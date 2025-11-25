#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Custom VisDrone dataset for YOLOX

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from .datasets_wrapper import Dataset


class VisdroneVIDDataset(Dataset):
    """
    VisDrone dataset class for YOLOX.
    Handles the specific directory structure of VisDrone dataset.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="imagenet_vid_train.json",
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        VisDrone dataset initialization. Annotation data are read into memory by COCO API.
        
        Args:
            data_dir (str): dataset root directory (e.g., /root/datasets/visdrone/yolov)
            json_file (str): COCO json file name
            name (str): dataset split name ('train' or 'val')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            cache (bool): whether to cache images in memory
        """
        super().__init__(img_size)
        
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser("~"), "datasets", "visdrone", "yolov")
        
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        
        # Load COCO annotations
        ann_file = os.path.join(self.data_dir, "annotations", self.json_file)
        self.coco = COCO(ann_file)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        
        self.annotations = self._load_coco_annotations()
        self.imgs = None
        
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        """Cache images into memory for faster training."""
        print("\nCaching images...")
        self.imgs = [None] * len(self.annotations)
        
        from tqdm import tqdm
        from multiprocessing.pool import ThreadPool
        
        NUM_THREADs = min(8, os.cpu_count())
        
        def load_img(index):
            img = self.load_resized_img(index)
            self.imgs[index] = img
            return index
        
        with ThreadPool(NUM_THREADs) as pool:
            results = list(tqdm(pool.imap(load_img, range(len(self.annotations))), 
                              total=len(self.annotations)))
        
        print(f"Cached {len(results)} images")

    def load_anno_from_ids(self, id_):
        """Load annotation from image ID."""
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        """Load image from file_name in annotation."""
        file_name = self.annotations[index][3]
        
        # Try multiple possible paths
        possible_paths = [
            # Direct path in root
            os.path.join(self.data_dir, file_name),
            # With train2017 symlink
            os.path.join(self.data_dir, "train2017", file_name),
            # With val2017 symlink
            os.path.join(self.data_dir, "val2017", file_name),
            # Original structure
            os.path.join(self.data_dir, "VisDrone-VID", file_name),
        ]
        
        img = None
        img_file = None
        
        for path in possible_paths:
            if os.path.exists(path):
                img_file = path
                img = cv2.imread(img_file)
                if img is not None:
                    break
        
        if img is None:
            raise FileNotFoundError(
                f"Could not load image: {file_name}\n"
                f"Tried paths:\n" + "\n".join(f"  - {p}" for p in possible_paths)
            )
        
        return img

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, resized_info, _ = self.annotations[index]
        
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        
        return img, target, img_info, img_id
