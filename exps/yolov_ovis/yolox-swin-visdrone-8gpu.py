#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX-Swin-Base VisDrone Config - Optimized for 8x GPU Training

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import random

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model architecture
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # VisDrone dataset configuration
        self.num_classes = 10
        self.data_dir = "/root/datasets/visdrone/yolov"
        self.train_ann = "imagenet_vid_train_coco.json"
        self.val_ann = "imagenet_vid_val_coco.json"
        
        # === OPTIMIZED FOR 8x GPU TRAINING ===
        # Extended training for better convergence
        self.max_epoch = 80
        self.no_aug_epochs = 10  # Last 10 epochs without augmentation
        self.warmup_epochs = 5
        self.eval_interval = 5  # Evaluate every 5 epochs (faster with 8 GPUs)
        
        # Learning rate optimized for large batch training
        # With 8 GPUs × 8 batch/gpu = 64 total batch
        # LR = base_lr * batch_size / 64
        self.basic_lr_per_img = 0.001 / 64.0  # 0.001 for Swin (lower than 0.005 for YOLOX-L)
        self.min_lr_ratio = 0.01
        
        # Scheduler and optimizer
        self.scheduler = "yoloxwarmcos"
        self.weight_decay = 0.05  # Higher for Transformer (vs 0.0005 for CNN)
        self.momentum = 0.9
        
        # === INPUT RESOLUTION ===
        # Option 1: Balanced resolution (default)
        self.input_size = (608, 1088)  # 16:9 aspect ratio
        self.test_size = (608, 1088)
        
        # Option 2: High resolution (uncomment for best accuracy)
        # self.input_size = (768, 1344)  # Native VisDrone resolution
        # self.test_size = (768, 1344)
        # Note: Reduce batch to 32 total (4 per GPU) for 768×1344
        
        # Multi-scale training
        self.multiscale_range = 6
        
        # === DATA AUGMENTATION (STRONG FOR AERIAL VIEWS) ===
        self.degrees = 15.0  # Rotation
        self.translate = 0.2  # Translation
        self.mosaic_scale = (0.5, 1.5)  # Scale range
        self.shear = 2.5
        
        # Mixup augmentation
        self.enable_mixup = True
        self.mixup_prob = 0.1  # Lower for Swin (more sensitive)
        self.mixup_scale = (0.8, 1.6)
        
        # Other augmentations
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.mosaic_prob = 1.0
        
        # === MULTI-GPU DATA LOADING ===
        self.data_num_workers = 8  # 8 workers per GPU = 64 total workers
        
        # === MODEL EMA ===
        self.ema = True  # Exponential moving average
        
        # === TEST CONFIGURATION ===
        self.test_conf = 0.001
        self.nmsthre = 0.5
        
        # === SWIN TRANSFORMER SPECIFIC ===
        self.backbone_name = "swin_base"
        self.pretrained_path = "pretrained/swin_base_patch4_window7_224_22k.pth"

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXSwinTransformer
        from yolox.models.yolo_head import YOLOXHead
        
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            
            # Swin-Base Transformer backbone
            backbone = YOLOXSwinTransformer(
                self.depth, 
                self.width, 
                in_channels=in_channels,
                arch='base',
                act=self.act
            )
            
            # YOLOX detection head
            head = YOLOXHead(
                self.num_classes, 
                self.width, 
                in_channels=in_channels, 
                act=self.act
            )
            
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            # Separate parameters: Transformer backbone vs detection head
            pg_transformer = []
            pg_head = []
            pg_bn = []
            
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg_bn.append(v.bias)
                
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg_bn.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if "backbone.backbone" in k:  # Swin Transformer layers
                        pg_transformer.append(v.weight)
                    else:  # Detection head
                        pg_head.append(v.weight)

            # Different weight decay for Transformer and head
            optimizer = torch.optim.SGD(
                [
                    {"params": pg_transformer, "weight_decay": self.weight_decay},  # 0.05 for Transformer
                    {"params": pg_head, "weight_decay": self.weight_decay * 0.1},   # 0.005 for head
                    {"params": pg_bn, "weight_decay": 0.0},  # No decay for BN
                ],
                lr=lr,
                momentum=self.momentum,
                nesterov=True,
            )
            self.optimizer = optimizer

        return self.optimizer

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
                    hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
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

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import ValTransform
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
            "batch_size": batch_size,
        }
        
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

    def eval(self, model, evaluator, is_distributed, half=False):
        """
        Evaluation with optional FP16 for faster inference
        """
        return evaluator.evaluate(model, is_distributed, half)
