#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX-L VisDrone Config v2 - Improved for better performance

import os
import torch.nn as nn
from yolox.exp import Exp as MyExp
import torch.distributed as dist
import torch
import random

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # YOLOX-L model size
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # VisDrone dataset configuration
        self.num_classes = 10  # VisDrone has 10 classes
        self.data_dir = "/root/datasets/visdrone/yolov"
        self.train_ann = "imagenet_vid_train_coco.json"
        self.val_ann = "imagenet_vid_val_coco.json"
        
        # === IMPROVED TRAINING CONFIGURATION ===
        # Extended training for better convergence
        self.max_epoch = 75  # 40 + 35 additional epochs
        self.no_aug_epochs = 10  # Increased from 5
        self.warmup_epochs = 5  # Increased from 2
        self.eval_interval = 1  # Validate after every epoch
        
        # Lower learning rate for stability
        self.min_lr_ratio = 0.01  # Decreased from 0.05
        self.basic_lr_per_img = 0.005 / 64.0  # Reduced from 0.01
        
        # === IMPROVED INPUT RESOLUTION ===
        # Higher resolution for better small object detection
        # Option 1: Balanced (current) - good speed/accuracy trade-off
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        
        # Option 2: High resolution (uncomment for best accuracy)
        # self.input_size = (768, 1344)  # Native VisDrone resolution
        # self.test_size = (768, 1344)
        # Note: This will be 2-3x slower but much better for small objects
        
        # Multi-scale training range
        self.multiscale_range = 6  # Increased from 5
        
        # === IMPROVED DATA AUGMENTATION ===
        # Stronger augmentation for aerial view variations
        self.degrees = 15.0  # Increased rotation from 10.0
        self.translate = 0.2  # Increased translation from 0.1
        self.mosaic_scale = (0.5, 1.5)  # More conservative than default (0.1, 2.0)
        self.shear = 2.5  # Increased from 2.0
        
        # Mixup augmentation
        self.enable_mixup = True
        self.mixup_prob = 0.15  # Probability of applying mixup
        self.mixup_scale = (0.8, 1.6)
        
        # HSV augmentation for varying lighting conditions
        self.hsv_prob = 1.0
        
        # Mosaic probability
        self.mosaic_prob = 1.0
        
        # === TEST CONFIGURATION ===
        self.test_conf = 0.001  # Confidence threshold
        self.nmsthre = 0.5  # NMS threshold
        
        # === EMA (Exponential Moving Average) ===
        self.ema = True  # Enable model EMA for better stability

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN
        from yolox.models.yolo_head import YOLOXHead
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            
            # Standard YOLOX-L backbone (CSPDarknet)
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.data.datasets.visdrone import VisdroneDataset
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = VisdroneDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name='train',
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import YoloBatchSampler, DataLoader, InfiniteSampler, ValTransform
        from yolox.data.datasets.visdrone import VisdroneDataset

        valdataset = VisdroneDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name='val',
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
