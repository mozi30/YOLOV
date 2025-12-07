#!/bin/bash
# YOLOX-Swin-Base Training Script for 8x GPU
# Optimized for VisDrone dataset

set -e

HOME_DIR=/root
cd $HOME_DIR/TemporalAttentionPlayground/YOLOV
source $HOME_DIR/miniconda3/etc/profile.d/conda.sh
conda activate yolox
pip install loguru
echo "Starting training..."
echo ""
while true; do
    python tools/vid_train.py \
    -n yolov_swinbase_window_9_new \
    -f /root/TemporalAttentionPlayground/YOLOV/exps/customed_example/yolov_swinbase_v2.py \
    --batch-size 8 \
    --fp16 \
    --resume \
    -c '/root/TemporalAttentionPlayground/YOLOV/V++_w8_gl_conf_01/yolov_swinbase_v2/latest_ckpt.pth'
done


echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase_window_9_new/"
echo "=========================================="
