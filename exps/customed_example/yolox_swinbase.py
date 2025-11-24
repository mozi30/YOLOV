#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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
        self.data_dir = "/home/mozi/datasets/visdrone/yolov"
        self.train_ann = "imagenet_vid_train_coco.json"
        self.val_ann = "imagenet_vid_val_coco.json"

        # -------------------------------
        # Training configuration (AP-oriented)
        # -------------------------------
        # Longer schedule with a no-aug tail
        self.max_epoch = 15
        self.warmup_epochs = 3
        self.no_aug_epochs = 5

        # Learning rate and optimizer
        self.act = "silu" # Activation function
        self.data_num_workers = 8 # Number of data loading workers
        self.basic_lr_per_img = 0.01 / 64
        self.min_lr_ratio = 0.01

        # Input size and multiscale training
        self.multiscale_range = 2
        self.input_size = (544, 960)
        self.test_size  = (544, 960)

        self.eval_interval = 1

        # -------------------------------
        # Augmentation / multiscale
        # -------------------------------
        self.mosaic_prob = 0.5   
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.hsv_prob = 0.5
        self.flip_prob = 0.5

        # geometric: slightly softened
        self.degrees = 5.0            
        self.translate = 0.1          
        self.shear = 2.0           
        
        # EMA (Exponential Moving Average)
        self.ema = True
        self.weight_decay = 1e-4

        self.test_conf = 0.01
        self.nmsthre = 0.2

        # Swin Transformer configuration
        self.backbone_name = "swin_base"  # Options: swin_tiny, swin_small, swin_base
        self.pretrained = True  # Use ImageNet pretrained weights
        self.pretrain_img_size = 224  # ImageNet pretrain size
        self.window_size = 7  # Swin window size

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN_Swin
        from yolox.models.yolo_head import YOLOXHead
        import torch.nn as nn

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            out_channels = [256, 512, 1024]
            backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                        out_channels=out_channels,
                                        act=self.act,
                                        in_features=(1, 2, 3),
                                        swin_depth=[2, 2, 18, 2],
                                        num_heads=[4, 8, 16, 32],
                                        base_dim=int(in_channels[0] / 2),
                                        pretrain_img_size=self.pretrain_img_size,
                                        window_size=self.window_size,
                                        width=self.width,
                                        depth=self.depth
                                        )

            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=out_channels,           # must match FPN out_channels
                act=self.act,
            )

            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model


    def random_resize(self, data_loader, epoch, rank, is_distributed):
        """
        Multiscale resize with a strict cap on the long edge (<= 1280).
        Locks to fixed input_size during the no-aug fine-tuning phase.
        """
        tensor = torch.LongTensor(2).cuda()
        if epoch >= (self.max_epoch - self.no_aug_epochs):
            return self.input_size  # (800, 1280)

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
                    max_labels=300,          # keep crowded-scene labels
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=300,          # keep crowded-scene labels
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
        from yolox.data import ValTransform
        from yolox.data.datasets.visdrone import VisdroneDataset

        valdataset = VisdroneDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

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
