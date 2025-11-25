#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import json
import os
import random

import numpy
from loguru import logger

import cv2
import numpy as np
import torch
#from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import Sampler,BatchSampler,SequentialSampler
from xml.dom import minidom
import math
from yolox.data.datasets import coco
from yolox.data.datasets.coco import remove_useless_info
from yolox.utils import xyxy2cxcywh

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]
XML_EXT = [".xml"]
name_list = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']
numlist = range(30)
name_num = dict(zip(name_list,numlist))


class VisDroneVID(torchDataset):
    """
    VisDrone-VID loader for the schema:

    {
      "info": {...},
      "videos": [
        {
          "id": int, "width": int, "height": int, "length": int,
          "file_names": [str, ... length items]
        }, ...
      ],
      "annotations": [
        {
          "id": int, "video_id": int, "category_id": int,
          "areas": [float or None, ... length items],
          "bboxes": [[x,y,w,h] or None, ... length items],
          "iscrowd": 0/1
        }, ...
      ] or None,
      "categories": [{"id": int, "name": str, "supercategory": str}, ...]
    }
    """

    def __init__(
        self,
        data_dir,
        json_file,          # path to the json described above
        name="train",       # split subfolder
        img_size=(640, 640),
        preproc=None,
        lframe=0,
        gframe=16,
        mode="random",      # "random" | "uniform" | "gl"
        val=False,
        tnum=3000
    ):
        super().__init__()
        self.data_dir = data_dir
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.lframe = lframe
        self.gframe = gframe
        self.mode = mode
        self.val = val
        self.tnum = tnum

        with open(json_file, "r") as f:
            jd = json.load(f)

        self.videos = jd.get("videos", [])
        self.annotations_raw = jd.get("annotations", None)
        cats = jd.get("categories", [])
        # contiguous class id map: cat_id -> [0..C-1]
        cat_ids_sorted = sorted([c["id"] for c in cats])
        self.catid2cid = {cid: i for i, cid in enumerate(cat_ids_sorted)}
        self.class_ids = cat_ids_sorted
        self._classes = tuple([c["name"] for c in sorted(cats, key=lambda x: x["id"])])

        # ---- build frame index and per-frame annotation map
        # frame_index: list of dicts with keys:
        #   {"key": (video_id, frame_idx), "file_name": str, "width": int, "height": int}
        self.frame_index = []
        # ann_map[(video_id, frame_idx)] -> list of dicts: {"bbox": [x1,y1,x2,y2], "cid": contiguous class id}
        self.ann_map = {}

        # Quick index: video_id -> video dict
        self.vid_by_id = {v["id"]: v for v in self.videos}

        for v in self.videos:
            W, H, L = v["width"], v["height"], v["length"]
            files = v["file_names"]
            assert len(files) == L, f"video {v['id']} file_names length mismatch: {len(files)} vs length {L}"
            for k in range(L):
                self.frame_index.append({
                    "key": (v["id"], k),
                    "file_name": files[k],
                    "width": W,
                    "height": H,
                })
                self.ann_map[(v["id"], k)] = []  # will fill below

        # Fill per-frame annotations if provided
        if self.annotations_raw is not None:
            for ann in self.annotations_raw:
                vid = ann["video_id"]
                cid = self.catid2cid[ann["category_id"]]
                bboxes = ann.get("bboxes", [])
                # areas optional; not needed for training
                L = self.vid_by_id[vid]["length"]
                # Be robust to missing/short arrays
                for k in range(min(L, len(bboxes))):
                    bb = bboxes[k]
                    if bb is None:
                        continue
                    x, y, w, h = bb
                    x1, y1 = max(0, x), max(0, y)
                    x2 = min(self.vid_by_id[vid]["width"],  x + max(0, w))
                    y2 = min(self.vid_by_id[vid]["height"], y + max(0, h))
                    if x2 >= x1 and y2 >= y1:
                        self.ann_map[(vid, k)].append({
                            "bbox": [x1, y1, x2, y2],
                            "cid": cid
                        })

        # ---- Build sequences of file_names for sampling modes (like your other datasets)
        # dataset_sequences is a list; each item is a list[str] (frame file paths for one video)
        self.dataset_sequences = []
        for v in self.videos:
            self.dataset_sequences.append(list(v["file_names"]))  # keep original order

        # ---- Build list of clips/res according to mode (like OVIS/Arg_VID)
        self.res = self.photo_to_sequence(self.lframe, self.gframe)

        # Optional: name->(video_id, frame_idx) mapping for quick index by file_name (if you need it)
        self.name_key_map = {}
        for fi in self.frame_index:
            self.name_key_map[fi["file_name"]] = fi["key"]

    def __len__(self):
        return len(self.res)

    # -------- sampling clips, same logic pattern as your classes
    def photo_to_sequence(self, lframe, gframe):
        res = []
        for seq in self.dataset_sequences:
            ele_len = len(seq)
            if ele_len < max(1, lframe + gframe):
                continue

            if self.mode == "random":
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

            elif self.mode == "uniform":
                split_num = ele_len // max(1, gframe)
                all_uniform = seq[:split_num * gframe]
                for i in range(split_num):
                    res.append(all_uniform[i::split_num])

            elif self.mode == "gl":
                split_num = ele_len // max(1, lframe)
                all_local = seq[:split_num * lframe]
                for i in range(split_num):
                    l_clip = all_local[i * lframe:(i + 1) * lframe]
                    others = seq[:i * lframe] + seq[(i + 1) * lframe:]
                    g_clip = random.sample(others, gframe) if gframe > 0 and len(others) >= gframe else []
                    res.append(l_clip + g_clip)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        if self.val:
            random.seed(42)
            random.shuffle(res)
            if self.tnum == -1:
                return res
            else:
                return res[:self.tnum]
        else:
            random.shuffle(res)
            if self.tnum == -1:
                return res[:15000]
            else:
                return res[:self.tnum]

    # -------- utilities
    def _resize_factor(self, H, W):
        return min(self.img_size[0] / H, self.img_size[1] / W)

    def _labels_for(self, file_name):
        """Return (rescaled_labels [N,5], img_info(h,w)) for a given frame file_name."""
        vid, k = self.name_key_map[file_name]
        v = self.vid_by_id[vid]
        H, W = v["height"], v["width"]
        r = self._resize_factor(H, W)

        anns = self.ann_map[(vid, k)]
        num = len(anns)
        labels = np.zeros((num, 5), dtype=np.float32)
        for i, a in enumerate(anns):
            x1, y1, x2, y2 = a["bbox"]
            labels[i, 0:4] = [x1 * r, y1 * r, x2 * r, y2 * r]
            labels[i, 4] = a["cid"]
        return labels, (H, W)

    # -------- PyTorch Dataset API compatible with your collate fns
    def pull_item(self, path):
        """
        path: relative file_name (one element from self.res's inner lists).
        Returns: img (resized), annos (rescaled [N,5]), img_info (H,W), returned_path (str)
        """
        # Absolute path
        #abs_path = os.path.join(self.data_dir, self.name, path)
        abs_path = os.path.join(self.data_dir, path)
        img = cv2.imread(abs_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {abs_path}")

        H, W = img.shape[:2]
        r = self._resize_factor(H, W)
        img_resized = cv2.resize(img, (int(W * r), int(H * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        annos, img_info = self._labels_for(path)
        return img_resized, annos, img_info, path

    def __getitem__(self, path):
        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.img_size)
        return img, target, img_info, path

class VIDDataset(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        file_path="train_seq.npy",
        img_size=(416, 416),
        preproc=None,
        lframe = 18,
        gframe = 6,
        val = False,
        mode='random',
        dataset_pth = '',
        tnum = 1000,
        formal = False,
        traj_linking = False,
        local_stride = 1
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.tnum = tnum
        self.traj_linking = traj_linking
        self.input_dim = img_size
        self.file_path = file_path
        self.mode = mode  # random, continous, uniform
        self.img_size = img_size
        self.preproc = preproc
        self.val = val
        self.formal = formal
        self.local_stride = local_stride
        self.res = self.photo_to_sequence(self.file_path,lframe,gframe)
        self.dataset_pth = dataset_pth

    def __len__(self):
        return len(self.res)


    def photo_to_sequence(self,dataset_path,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frames in a video dir
        Returns:
            split result
        '''
        res = []
        dataset = np.load(dataset_path,allow_pickle=True).tolist()
        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                if self.formal:
                    res.append(element)
                else:
                    continue
                # res.append(element)
                # continue
            else:
                if self.mode == 'random':
                    if lframe == 0:
                        split_num = int(ele_len / (gframe))
                        random.shuffle(element)
                        for i in range(split_num):
                            res.append(element[i * gframe:(i + 1) * gframe])
                        if self.formal and len(element[split_num * gframe:]):
                            tail = element[split_num * gframe:]
                            # padding = tail + element[:gframe-len(tail)]
                            res.append(tail)
                    elif lframe!=0:
                        if self.local_stride==1:
                            split_num = int(ele_len / (lframe))
                            all_local_frame = element[:split_num * lframe]
                            for i in range(split_num):
                                if self.traj_linking and i!=0:
                                    l_frame = all_local_frame[i * lframe-1:(i + 1) * lframe]
                                else:
                                    l_frame = all_local_frame[i * lframe:(i + 1) * lframe]
                                g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                                res.append(l_frame + g_frame)
                            if self.formal and len(element[split_num * lframe:]):
                                if self.traj_linking:
                                    tail = element[split_num * lframe-1:]
                                else:
                                    tail = element[split_num * lframe:]
                                res.append(tail)
                        else:
                            split_num = ele_len//(lframe*self.local_stride)
                            for i in range(split_num):
                                for j in range(self.local_stride):
                                    res.append(element[lframe * self.local_stride * i:lframe * self.local_stride * (i + 1)][
                                                   j::self.local_stride])
                    else:
                        print('unsupport mode, exit')
                        exit(0)

                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])

                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            if self.tnum == -1:
                return res
            else:
                return res[:self.tnum]
        else:
            random.shuffle(res)
            return res[:15000]


    def get_annotation(self,path,test_size):
        path = path.replace("Data","Annotations").replace("JPEG","xml")
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        for xmls in files:
            photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
            file = minidom.parse(xmls)
            root = file.documentElement
            objs = root.getElementsByTagName("object")
            width = int(root.getElementsByTagName('width')[0].firstChild.data)
            height = int(root.getElementsByTagName('height')[0].firstChild.data)
            tempnode = []
            for obj in objs:
                nameNode = obj.getElementsByTagName("name")[0].firstChild.data
                xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
                xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
                ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
                ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
                x1 = np.max((0,xmin))
                y1 = np.max((0,ymin))
                x2 = np.min((width,xmax))
                y2 = np.min((height,ymax))
                if x2 >= x1 and y2 >= y1:
                    #tempnode.append((name_num[nameNode],x1,y1,x2,y2,))
                    tempnode.append(( x1, y1, x2, y2,name_num[nameNode],))
            num_objs = len(tempnode)
            res = np.zeros((num_objs, 5))
            r = min(test_size[0] / height, test_size[1] / width)
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]
            res[:, :-1] *= r
            anno_res.append(res)
        return anno_res


    def pull_item(self,path):
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
        path = os.path.join(self.dataset_pth,path)
        annos = self.get_annotation(path, self.img_size)[0]

        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos, img_info, path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path

class Arg_VID(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        data_dir='/media/tuf/ssd/Argoverse-1.1/',
        img_size=(416, 640),
        preproc=None,
        lframe = 0,
        gframe = 16,
        val = False,
        mode='random',
        COCO_anno = '',
        name = "tracking",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.input_dim = img_size
        self.name = name
        self.val = val
        self.data_dir = data_dir
        self.img_size = img_size
        self.coco_anno_path = COCO_anno
        self.name_id_dic = self.get_NameId_dic()
        self.coco = coco(COCO_anno)
        remove_useless_info(self.coco)
        self.ids = sorted(self.coco.getImgIds())
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.mode = mode  # random, continous, uniform
        self.preproc = preproc

        self.res = self.photo_to_sequence(lframe,gframe)

    def get_NameId_dic(self):
        img_dic = {}
        with open(self.coco_anno_path,'r') as train_anno_content:
            train_anno_content = json.load(train_anno_content)
            for im in train_anno_content['images']:
                img_dic[im['name']] = im['id']
        return img_dic

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def __len__(self):
        return len(self.res)

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_ann['name'] = self.coco.dataset['seq_dirs'][im_ann['sid']] + '/' + im_ann['name']
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

        file_name = (
            im_ann["name"]
            if "name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def photo_to_sequence(self,lframe,gframe, seq_len = 192):
        '''

        Args:
            dataset_path: list,every element is a list contain all frame in a video dir
        Returns:
            split result
        '''
        res = []

        with open(self.coco_anno_path, 'r') as anno:
            anno = json.load(anno)
            dataset = [[] for i in range(len(anno['sequences']))]
            for im in anno['images']:
                dataset[im['sid']].append(self.coco.dataset['seq_dirs'][im['sid']] + '/' + im['name'])
            for ele in dataset:
                sorted(ele)

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    # split_num = int(ele_len / (gframe))
                    # random.shuffle(element)
                    # for i in range(split_num):
                    #     res.append(element[i * gframe:(i + 1) * gframe])
                    # if self.val and element[(i + 1) * gframe:] != []:
                    #     res.append(element[(i + 1) * gframe:])

                    seq_split_num = int(len(element) / seq_len)
                    for k in range(seq_split_num + 1):
                        tmp = element[k * seq_len:(k + 1) * seq_len]
                        if tmp == []:continue
                        random.shuffle(tmp)
                        split_num = int(len(tmp) / (gframe))
                        for i in range(split_num):
                            res.append(tmp[i * gframe:(i + 1) * gframe])
                        if self.val and tmp[(i + 1) * gframe:] != []:
                            res.append(tmp[(i + 1) * gframe:])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            # random.seed(42)
            # random.shuffle(res)
            return res#[:1000]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res#[:1000]#[:15000]


    def pull_item(self,path):
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
        path = path.split('/')[-1]
        idx = self.name_id_dic[path]
        annos, img_info, resized_info, img_path = self.annotations[idx]
        abs_path = os.path.join(self.data_dir, self.name, img_path)
        img = cv2.imread(abs_path)

        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path


class OVIS(Arg_VID):
    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        #im_ann['name'] = self.coco.dataset['seq_dirs'][im_ann['sid']] + '/' + im_ann['name']
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

        file_name = (
            im_ann["name"]
            if "name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def photo_to_sequence(self,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frame in a video dir
        Returns:
            split result
        '''
        res = []

        with open(self.coco_anno_path, 'r') as anno:
            anno = json.load(anno)
            dataset = [[] for i in range(len(anno['videos']))]
            for im in anno['images']:
                dataset[im['sid']].append(im['name'])
            for ele in dataset:
                sorted(ele)

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    split_num = int(ele_len / (gframe))
                    random.shuffle(element)
                    for i in range(split_num):
                        res.append(element[i * gframe:(i + 1) * gframe])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            random.seed(42)
            random.shuffle(res)
            return res#[2000:3000]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res#[:15000]

    def pull_item(self,path):
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
        idx = self.name_id_dic[path]
        annos, img_info, resized_info, img_path = self.annotations[idx]
        abs_path = os.path.join(self.data_dir,self.name, img_path)
        img = cv2.imread(abs_path)

        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_path



def get_xml_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in XML_EXT:
                image_names.append(apath)

    return image_names

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def make_path(train_dir,save_path):
    res = []
    for root,dirs,files in os.walk(train_dir):
        temp = []
        for filename in files:
            apath = os.path.join(root, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                temp.append(apath)
        if(len(temp)):
            temp.sort()
            res.append(temp)
    res_np = np.array(res,dtype=object)
    np.save(save_path,res_np)


class TestSampler(SequentialSampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class TrainSampler(Sampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.data_source.res)
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class VIDBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            for filename in ele:
                batch.append(filename)
                if (len(batch)) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch)>0 and not self.drop_last:
            yield batch
    def __len__(self):
        return len(self.sampler)

class VIDBatchSampler_Test(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            yield ele
            # for filename in ele:
            #     batch.append(filename)
            #     if (len(batch)) == self.batch_size:
            #         yield batch
            #         batch = []
            # if len(batch)>0 and not self.drop_last:
            #     yield batch
    def __len__(self):
        return len(self.sampler)

def collate_fn(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([300,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        #path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    # path_sequence= torch.tensor(path_sequence)
    # time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,None

def get_vid_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

def vid_val_loader(batch_size,data_num_workers,dataset,):
    sampler = VIDBatchSampler_Test(TestSampler(dataset),batch_size,drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': collate_fn
    }
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return loader

def collate_fn_trans(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([100,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(copy.deepcopy(sample[1])))
        sample[1][:,1:]=xyxy2cxcywh(sample[1][:,1:])
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    path_sequence= torch.tensor(path_sequence)
    time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,time_embedding

def get_trans_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.max_iter = len(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target,_,_,_,self.time_ebdding = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.time_ebdding = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        time_ebdding = self.time_ebdding
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target,time_ebdding

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

def get_timing_signal_1d(index_squence,channels,min_timescale=1.0, max_timescale=1.0e4,):
    num_timescales = channels // 2

    log_time_incre = torch.tensor(math.log(max_timescale/min_timescale)/(num_timescales-1))
    inv_timescale = min_timescale*torch.exp(torch.arange(0,num_timescales)*-log_time_incre)

    scaled_time = torch.unsqueeze(index_squence,1)*torch.unsqueeze(inv_timescale,0) #(index_len,1)*(1,channel_num)
    sig = torch.cat([torch.sin(scaled_time),torch.cos(scaled_time)],dim=1)
    return sig
