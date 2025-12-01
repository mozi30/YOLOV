#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Custom VisDrone dataset for YOLOX

import os
import random
import cv2
import numpy as np
from pycocotools.coco import COCO

from torch.utils.data.dataset import Dataset as torchDataset
from .datasets_wrapper import Dataset


VISDRONE_CATEGORIES = [
    {"id": 1,  "name": "pedestrian",        "supercategory": "person"},
    {"id": 2,  "name": "people",            "supercategory": "person"},
    {"id": 3,  "name": "bicycle",           "supercategory": "vehicle"},
    {"id": 4,  "name": "car",               "supercategory": "vehicle"},
    {"id": 5,  "name": "van",               "supercategory": "vehicle"},
    {"id": 6,  "name": "truck",             "supercategory": "vehicle"},
    {"id": 7,  "name": "tricycle",          "supercategory": "vehicle"},
    {"id": 8,  "name": "awning-tricycle",   "supercategory": "vehicle"},
    {"id": 9,  "name": "bus",               "supercategory": "vehicle"},
    {"id": 10, "name": "motor",             "supercategory": "vehicle"},
    {"id": 11, "name": "others",            "supercategory": "unknown"}
]


class VidDroneVIDataset(torchDataset):
    def __init__(
        self,
        data_dir,
        split = "train",
        img_size=(640, 640),
        preproc=None,
        lframe = 0,
        gframe = 0,
        sample_mode = "random", # "random" or "uniform" | "gl"
        max_epoch_samples = -1,
        
        # New Settings
        gl_stride = 1,
        ignore_regions = False,
        pertubations = None,
        pertubations_seed = 42, # If -1 -> random seed
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.split_folder = "VisDrone-VID-" + self.split
        self.img_size = img_size
        self.preproc = preproc
        self.lframe = lframe
        self.gframe = gframe
        self.sample_mode = sample_mode
        self.max_epoch_samples = max_epoch_samples
        self.ignore_regions = ignore_regions
        self.pertubations = pertubations
        self.pertubations_seed = pertubations_seed
        self.annotations = self.build_dataset_from_directory()
        self.gl_stride = max(1, gl_stride)
        if( self.split == "val"):
            self.val = True
        else:
            self.val = False

        assert gframe + lframe > 0, "gframe and lframe cannot be both zero."

        self.videos = []

        for video_ann in self.annotations["videos"]:
            vid_id = video_ann["video_id"]
            width = video_ann["width"]
            height = video_ann["height"]

            # collect all frame names for this video in order
            file_names = [img["file_name"] for img in self.annotations["video_images"][vid_id]]
            image_uid = [(vid_id, img["image_id"]) for img in self.annotations["video_images"][vid_id]]
            self.videos.append({
                "id": vid_id,
                "width": width,
                "height": height,
                "length": len(file_names),
                "file_names": file_names,
                "image_uid": image_uid
            })
        
        self.dataset_sequences = []
        for v in self.videos:
            # one sequence per video, list of frame file names in order
            self.dataset_sequences.append(list(v["image_uid"]))

        self.res = self.photo_to_sequence(self.lframe, self.gframe)

        # Flat frame index and per-frame annotation map
        self.frame_index = []          # list of {"key": (vid_id, frame_idx), "file_name": ..., "width": ..., "height": ...}
        self.ann_map = {}              # (vid_id, frame_idx) -> list of {"bbox": [x1,y1,x2,y2], "cid": class_id}

        # Optional: category-id → contiguous-id map
        cats = self.annotations["categories"]
        cat_ids_sorted = sorted([c["id"] for c in cats])
        self.catid2cid = {cid: i for i, cid in enumerate(cat_ids_sorted)}

        # Quick lookup: video_id → image_annotations dict
        video_ann_list = self.annotations["video_annotations"]  # list of {"video_id": vid, "annotations": {image_id: [...]}}
        vid_to_img_ann = {va["video_id"]: va["annotations"] for va in video_ann_list}

        for v in self.videos:
            vid = v["id"]
            width = v["width"]
            height = v["height"]

            # images list for this video (already ordered by frame_id in your builder)
            images = self.annotations["video_images"][vid]  # list of image dicts
            img_ann = vid_to_img_ann.get(vid, {})

            for frame_idx, img_info in enumerate(images):
                file_name = img_info["file_name"]
                image_id = img_info["image_id"]

                key = (vid, frame_idx)

                # 1) frame_index entry
                self.frame_index.append({
                    "key": key,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                })

                # 2) annotations for this frame
                self.ann_map[key] = []
                for ann in img_ann.get(image_id, []):
                    x1, y1, x2, y2 = ann["bbox"]  # already xyxy
                    cid = self.catid2cid[ann["category_id"]]
                    self.ann_map[key].append({
                        "bbox": [x1, y1, x2, y2],
                        "cid": cid,
                    })
    


    def get_video_size(self, video_name):            
        seq_path = os.path.join(self.data_dir, self.split_folder, "sequences", video_name)
        img_files = [f for f in os.listdir(seq_path) if f.endswith('.jpg') or f.endswith('.png')]
        if not img_files:
            raise FileNotFoundError(f"No image files found in {seq_path}")
        first_img_path = os.path.join(seq_path, img_files[0])
        img = cv2.imread(first_img_path)
        if img is None:
            raise ValueError(f"Could not read image file {first_img_path}")
        return img.shape[1], img.shape[0]  # width, height

    def build_dataset_from_directory(self):
        #check if directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
        if not os.path.exists(os.path.join(self.data_dir, self.split_folder)):
            raise FileNotFoundError(f"Split folder {self.split_folder} does not exist in {self.data_dir}.")
        if not os.path.exists(os.path.join(self.data_dir,self.split_folder, "annotations")):
            raise FileNotFoundError(f"Annotations folder does not exist in {self.data_dir}.")
        if not os.path.exists(os.path.join(self.data_dir, self.split_folder, "sequences")):
            raise FileNotFoundError(f"Sequences folder does not exist in {self.data_dir}.")
        
        annotations = {}
        videos = []
        video_images = {}
        video_annotations = {}
        video_ignored_regions = {}
        video_id = 0
        for annotation_file in os.listdir(os.path.join(self.data_dir, self.split_folder, "annotations")):
            if annotation_file.endswith(".txt"):
                annotation_path = os.path.join(self.data_dir,self.split_folder, "annotations", annotation_file)
                video_name = annotation_file[:-4]  # Remove .txt extension
                width, height = self.get_video_size(video_name)
                videos.append({"video_id": video_id, "video_name": video_name,
                                "video_path": os.path.join(self.split_folder, "sequences", video_name),
                                  "width": width, "height": height})
                video_images[video_id] = []
                image_ignored_regions = {}
                image_annotations = {}
                image_id = 0
                seen_images = {}
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(',')
                        frame_id = int(parts[0])
                        target_id = int(parts[1])
                        def xywh_to_xyxy(x, y, w, h):
                            x1 = x
                            y1 = y
                            x2 = x + w
                            y2 = y + h
                            return [x1, y1, x2, y2]
                        bbox = xywh_to_xyxy(float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                        score = float(parts[6])
                        category_id = int(parts[7])
                        trucated = int(parts[8])
                        occluded = int(parts[9])

                        image_name = f"{frame_id:06d}.jpg"
                        id_of_image = None
                        if image_name not in seen_images:
                            video_images[video_id].append({
                                "image_id": image_id,
                                "file_name": image_name,
                                "width": width,
                                "height": height,
                                "frame_id": frame_id,
                                "video_id": video_id,
                            })
                            seen_images[image_name] = image_id
                            id_of_image = image_id
                            image_id += 1
                        else:
                            id_of_image = seen_images[image_name]

                        if(category_id == 0):  # Assuming 0 is the ignore region class
                             if id_of_image not in image_ignored_regions:
                                image_ignored_regions[id_of_image] = []
                             image_ignored_regions[id_of_image].append(bbox)
                             continue
                        
                        if id_of_image not in image_annotations:
                            image_annotations[id_of_image] = []
                        image_annotations[id_of_image].append({
                            "bbox": bbox,
                            "score": score,
                            "category_id": category_id,
                            "truncated": trucated,
                            "occluded": occluded,
                            "target_id": target_id
                        })
                        
                video_ignored_regions.append({"video_id": video_id, "ignored_regions": image_ignored_regions})
                video_annotations.append({"video_id": video_id, "annotations": image_annotations})
        
                video_id += 1
        annotations["videos"] = videos
        annotations["video_images"] = video_images
        annotations["video_annotations"] = video_annotations
        annotations["video_ignored_regions"] = video_ignored_regions
        annotations["categories"] = VISDRONE_CATEGORIES

        return annotations
    
    def photo_to_sequence(self, lframe, gframe):
        res = []
        for seq in self.dataset_sequences:
            ele_len = len(seq)
            if ele_len < max(1, lframe + gframe):
                continue

            if self.sample_mode == "random":
                if lframe == 0:
                    split_num = ele_len // gframe
                    tmp = seq[:]  # avoid in-place shuffle of source
                    random.shuffle(tmp)
                    for i in range(split_num):
                        res.append(tmp[i * gframe:(i + 1) * gframe])
                    tail = tmp[split_num * gframe:]
                    if self.val and len(tail):
                        res.append(tail)
                else:
                    split_num = ele_len // lframe
                    all_local = seq[:split_num * lframe]
                    for i in range(split_num):
                        l_clip = all_local[i * lframe:(i + 1) * lframe]
                        others = seq[:i * lframe] + seq[(i + 1) * lframe:]
                        g_clip = random.sample(others, gframe) if gframe > 0 and len(others) >= gframe else []
                        res.append(l_clip + g_clip)

            elif self.sample_mode == "uniform":
                split_num = ele_len // max(1, gframe)
                all_uniform = seq[:split_num * gframe]
                for i in range(split_num):
                    res.append(all_uniform[i::split_num])

            elif self.sample_mode == "gl":
                split_num = ele_len // max(1, lframe)
                all_local = seq[:split_num * lframe]
                # use stride over local clips
                for i in range(0, split_num, self.gl_stride):
                    l_clip = all_local[i * lframe:(i + 1) * lframe]
                    others = seq[:i * lframe] + seq[(i + 1) * lframe:]
                    g_clip = random.sample(others, gframe) if gframe > 0 and len(others) >= gframe else []
                    res.append(l_clip + g_clip)
            else:
                raise ValueError(f"Unsupported mode: {self.sample_mode}")

        if self.val:
            random.seed(42)
            random.shuffle(res)
            if self.max_epoch_samples == -1:
                return res
            else:
                return res[:self.max_epoch_samples]
        else:
            random.shuffle(res)
            if self.max_epoch_samples == -1:
                return res[:15000]
            else:
                return res[:self.max_epoch_samples]
    

    def _resize_factor(self, H, W):
        return min(self.img_size[0] / H, self.img_size[1] / W)

    def _labels_for(self, vid, frame_idx):
        v = next(v for v in self.videos if v["id"] == vid)
        H, W = v["height"], v["width"]
        r = self._resize_factor(H, W)

        anns = self.ann_map.get((vid, frame_idx), [])
        num = len(anns)
        labels = np.zeros((num, 5), dtype=np.float32)
        for i, a in enumerate(anns):
            x1, y1, x2, y2 = a["bbox"]
            labels[i, 0:4] = [x1 * r, y1 * r, x2 * r, y2 * r]
            labels[i, 4] = a["cid"]
        return labels, (H, W)

    def pull_item(self, key):
        """
        key: (video_id, frame_idx) from self.res / dataset_sequences.
        """
        vid, frame_idx = key

        # image meta
        img_infos = self.annotations["video_images"][vid]
        img_info_dict = img_infos[frame_idx]
        file_name = img_info_dict["file_name"]           # e.g. "000123.jpg"

        video_path = next(v["video_path"] for v in self.annotations["videos"]
                        if v["video_id"] == vid)

        # datadir + video_path + file_name
        abs_path = os.path.join(self.data_dir, video_path, file_name)
        img = cv2.imread(abs_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {abs_path}")

        H, W = img.shape[:2]
        r = self._resize_factor(H, W)
        img_resized = cv2.resize(
            img,
            (int(W * r), int(H * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        annos, img_info = self._labels_for(vid, frame_idx)
        return img_resized, annos, img_info, key
            
    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.img_size)
        return img, target, img_info,path               
                        


class VisdroneDataset(Dataset):
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
