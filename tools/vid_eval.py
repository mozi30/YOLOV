#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from yolox.utils.logger import setup_logger
from loguru import logger
import os
    
import torch
import torch.backends.cudnn as cudnn

from yolox.data.datasets.visdrone import PerturbSpec, PerturbationSettings, PerturbationType, Severity, VidDroneVIDataset
from yolox.core import launch
from yolox.core.vid_trainer import Trainer

from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices
from yolox.data.data_augment import Vid_Val_Transform
from yolox.data.datasets import vid

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--tsize", default=576, type=int, help="test img size")
    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-urlT",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default='', type=str, help="checkpoint file")
    parser.add_argument(
        '-data_dir',
        default='',
        type=str,
        help="path to your dataset",

    )
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--perturbation', default=False, action="store_true",help='use perturbation for eval')
    parser.add_argument('--select_perturbation', default=PerturbationType.GAUSSIAN_NOISE, help='select perturbation type for eval')
    parser.add_argument('--severity', default=Severity.LOW, help='perturbation severity for eval')

    parser.add_argument('--lframe', default=0,type=int, help='local frame num')
    parser.add_argument('--gframe', default=32,type=int, help='global frame num')
    parser.add_argument('--mode', default='random', help='frame sample mode')
    parser.add_argument('--tnum', default=-1, help='vid test sequences')
    parser.add_argument('--formal', default=False, action="store_true",help='vid test sequences')
    parser.add_argument('--stride', default=1,type=int, help='global-local frame stride for eval')
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    lframe = int(args.lframe)
    gframe = int(args.gframe)
    stride = int(args.stride)

    # dataset_val = vid.VIDDataset(file_path='./yolox/data/datasets/val_seq.npy',
    #                              img_size=(args.tsize, args.tsize), preproc=Vid_Val_Transform(), lframe=lframe,
    #                              gframe=gframe, val=True,mode=args.mode,dataset_pth=exp.data_dir,tnum=int(args.tnum),
    #                              formal=args.formal,local_stride=exp.local_stride,)
    # val_loader = vid.vid_val_loader(batch_size=lframe + gframe, data_num_workers=4, dataset=dataset_val,)

    #val_loader = exp.get_eval_loader(batch_size=args.batch_size)

    ##  customed dataset here:
    # dataset_val = vid.OVIS(data_dir='/opt/dataset/OVIS', img_size=exp.test_size, mode='random',
    #                        COCO_anno='/opt/dataset/OVIS/ovis_train.json', name='train',
    #                        lframe=0, gframe=gframe, preproc=Vid_Val_Transform()
    #                        )
    # val_loader = vid.vid_val_loader(batch_size=lframe + gframe, data_num_workers=4, dataset=dataset_val, )

    assert lframe + gframe == args.batch_size, "Error: lframe + gframe should be equal to batch_size!!!"
    exp.gframe_val = gframe
    exp.lframe_val = lframe
    assert exp.lframe_val + exp.gframe_val == args.batch_size, "Error: exp.lframe_val + exp.gframe_val should be equal to batch_size!!!"
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    setup_logger(
        file_name,
        distributed_rank=0,
        filename="train_log.txt",
        mode="a",
    )
    logger.info("Building dataset...")
    logger.info("Val gframe: {}, lframe: {}".format(exp.gframe_val, exp.lframe_val))
    logger.info("Val stride: {}".format(stride))
    logger.info("Val batch size: {}".format(args.batch_size))


    if args.perturbation:
        selected_perturbation = args.select_perturbation
        severity = args.severity


        print("Using perturbation for eval!")
        logger.info("Perturbation eval: {}".format(args.perturbation))
        logger.info("Selected perturbation: {}".format(args.select_perturbation))
        logger.info("Perturbation severity: {}".format(args.severity))
        logger.info("Perturbation probability: {}".format(0.5))
        # perturb = PerturbationSettings(
        #     enabled=True,
        #     seed=123,
        #     shuffle_order=True,
        #     specs=[
        #         PerturbSpec(selected_perturbation, active=True, severity=severity, p=1),
        #     ],
        # )
        perturb = PerturbationSettings(
            enabled=True,
            seed=123,
            shuffle_order=True,
            specs=[
                PerturbSpec(PerturbationType.GAUSSIAN_NOISE, active=True,  severity=Severity.HIGH,  p=0.082),
                PerturbSpec(PerturbationType.MOTION_BLUR,    active=True, severity=Severity.HIGH,  p=0.082),
                PerturbSpec(PerturbationType.JPEG_COMPRESSION,active=True, severity=Severity.HIGH, p=0.082),
                PerturbSpec(PerturbationType.BRIGHTNESS_CHANGE,active=True, severity=Severity.HIGH,  p=0.082),
                PerturbSpec(PerturbationType.CONTRAST_CHANGE,  active=True, severity=Severity.HIGH,  p=0.082),
                PerturbSpec(PerturbationType.PIXELATION,     active=True, severity=Severity.HIGH,  p=0.082),
                PerturbSpec(PerturbationType.DEFOCUS_BLUR,    active=True, severity=Severity.HIGH,  p=0.082),
            ],
        )
    else:
        perturb = PerturbationSettings(enabled=False)



    
    
    
    dataset_val = VidDroneVIDataset(
            data_dir=exp.data_dir,
            split="val",
            img_size=exp.test_size,
            preproc=Vid_Val_Transform(),
            lframe=exp.lframe_val,
            gframe=exp.gframe_val,
            sample_mode="gl",
            max_epoch_samples=500,
            gl_stride = stride,
            perturb=perturb,
        )

    val_loader = vid.vid_val_loader(batch_size=exp.lframe_val + exp.gframe_val, data_num_workers=1, dataset=dataset_val, )


    print("Gframe and Lframe for eval: ", exp.gframe_val, exp.lframe_val)
    trainer = Trainer(exp, args, val_loader, val=True)


if __name__ == "__main__":
    args = make_parser().parse_args()

    exp = get_exp(args.exp_file, args.name)


    if args.lframe != None: exp.lframe_val = int(args.lframe)
    if args.gframe != None: exp.gframe_val = int(args.gframe)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    args.machine_rank = 1
    dist_url = "auto" #if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
