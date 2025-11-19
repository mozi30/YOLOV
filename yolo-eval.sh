#!/bin/bash
# YOLOX-Swin-Base Training Script for 8x GPU
# Optimized for VisDrone dataset

set -e

HOME_DIR=/home/mozi
cd $HOME_DIR/TemporalAttentionPlayground/YOLOV
source $HOME_DIR/miniconda3/etc/profile.d/conda.sh
conda activate yolox

echo "Starting eval..."
echo ""

python tools/eval.py \
  -f exps/customed_example/yolox_swinbase.py \
  -c YOLOX_outputs/yolox_swinbase/latest_ckpt.pth \
  -b 2 \
  -d 1 \
  --conf 0.001 \
  --nms 0.7 \
  --fp16 \
  --fuse

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
