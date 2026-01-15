#!/bin/bash
# YOLOX-Swin-Base Training Script for 8x GPU
# Optimized for VisDrone dataset

set -e

HOME_DIR=/root
cd $HOME_DIR/TemporalAttentionPlayground/YOLOV
source $HOME_DIR/miniconda3/etc/profile.d/conda.sh
conda activate yolox

echo "Starting eval..."
echo ""
# perturbations2=(
#   gaussian_noise
#   motion_blur
#   jpeg_compression
#   brightness_change
#   contrast_change
#   pixelation
#   defocus_blur
# )

# severities=(low med high)

# for p in "${perturbations2[@]}"; do
#     for s in "${severities[@]}"; do
python tools/eval.py \
  -f exps/customed_example/yolox_swinbase.py\
  -c /root/weights/yolov/yolox-swinbase_w7.pth\
  -b 8 \
  -d 1 \
  --fp16 \
  --fuse \
  --perturbation \
      # --select_perturbation $p \
      # --severity $s
#   done
# done





echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
