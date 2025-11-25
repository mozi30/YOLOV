#!/bin/bash
# YOLOX-Swin-Base Training Script for 8x GPU
# Optimized for VisDrone dataset

set -e

HOME_DIR=/home/mozi
cd $HOME_DIR/TemporalAttentionPlayground/YOLOV
source $HOME_DIR/miniconda3/etc/profile.d/conda.sh
conda activate yolox

echo "Starting training..."
echo ""
python3 tools/vid_train.py \
    -n yolov_swinbase_window_2 \
    -f /home/mozi/TemporalAttentionPlayground/YOLOV/exps/customed_example/yolov_swinbase.py \
    --batch-size 2 \
    --fp16 \
    -c $HOME_DIR/models/yolox-swinbase/best_ckpt.pth

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
