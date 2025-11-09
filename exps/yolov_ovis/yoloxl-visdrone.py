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
        self.data_dir = "/root/datasets/visdrone/yolov"
        self.train_ann = "imagenet_vid_train_coco.json"
        self.val_ann = "imagenet_vid_val_coco.json"

        # -------------------------------
        # Training configuration (AP-oriented)
        # -------------------------------
        # Longer schedule with a no-aug tail
        self.max_epoch = 110
        self.warmup_epochs = 5
        self.no_aug_epochs = 22            # last 22 epochs are no-aug

        # LR scaling rule: lr = basic_lr_per_img * total_batch_size
        # Slight bump vs default to aid convergence on tiny objects
        self.min_lr_ratio = 0.01
        self.basic_lr_per_img = 0.006 / 64.0

        # Use L1 (YOLOX L1 regression head). If your trainer supports toggling,
        # it’s best enabled only during the no-aug tail; otherwise keeping it on is fine.
        self.use_l1 = True

        self.eval_interval = 1

        # -------------------------------
        # Augmentation / multiscale
        # -------------------------------
        # Keep small objects: narrower shrink, wider range of scales
        self.multiscale_range = 5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.9, 1.3)     # protect small objects (less shrinking)
        self.shear = 2.0

        # MixUp off for sharper small-object boundaries
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.mixup_scale = (0.8, 1.6)

        # HSV and mosaic augmentation
        self.hsv_prob = 1.0
        self.mosaic_prob = 1.0

        # EMA (Exponential Moving Average)
        self.ema = True

        # -------------------------------
        # Input sizes (≤ 1300 long edge)
        # -------------------------------
        # Keep H,W divisible by 32. Landscape 16:9-ish, long edge <= 1280.
        self.input_size = (800, 1280)   # train target (H, W)
        self.test_size  = (832, 1280)   # eval target (slightly larger H, same W)

        # Inference / eval thresholds
        self.test_conf = 0.001
        self.nmsthre = 0.62             # slightly tighter NMS for crowded scenes

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
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
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

        # Lock to fixed input during no-aug phase
        if epoch >= (self.max_epoch - self.no_aug_epochs):
            # Keep exactly the configured training input size for determinism
            return self.input_size  # (800, 1280)

        if rank == 0:
            # size_factor = W/H based on configured input_size
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                base_tiles = int(self.input_size[0] / 32)
                smin = max(1, base_tiles - self.multiscale_range)
                smax = base_tiles + self.multiscale_range
                self.random_size = (smin, smax)

            smin, smax = self.random_size

            # Ensure the long edge does not exceed 1280
            # Long edge tiles cap
            max_long_tiles = 1280 // 32  # 40
            # If landscape (size_factor > 1), width is long edge: W = 32 * idx * size_factor
            # -> idx <= max_long_tiles / size_factor
            # If portrait-ish, height is the long edge: idx <= max_long_tiles
            idx_cap = int(max_long_tiles / max(1.0, size_factor))
            smax = min(smax, max(smin, idx_cap))

            # Bias to larger sizes within the allowed range
            size_idx = random.randint((smin + smax + 1) // 2, smax)
            h = 32 * size_idx
            w = 32 * int(size_idx * size_factor)

            # Final safety: clamp width to 1280 if rounding pushes it over
            if w > 1280:
                w = 1280
                # maintain /32 constraint for height as well
                h = 32 * int(round(w / size_factor / 32))

            tensor[0] = h
            tensor[1] = w

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
