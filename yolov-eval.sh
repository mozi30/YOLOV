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
# python tools/vid_eval.py \
#   -f exps/customed_example/yolov_swinbase.py\
#   -c /root/TemporalAttentionPlayground/YOLOV/V++_outputs/yolov_swinbase/best_ckpt.pth\
#   -b 2 \
#   -d 1 \
#   --fp16 \
#   --lframe 2 \
#   --gframe 0 \

# python tools/vid_eval.py \
#   -f exps/customed_example/yolov_swinbase_v2.py\
#   -c /root/TemporalAttentionPlayground/YOLOV/V++_new_outputs/yolov_swinbase_v2/best_ckpt.pth\
#   -b 4 \
#   -d 1 \
#   --fp16 \
#   --lframe 0 \
#   --gframe 4 \

# python tools/vid_eval.py \
#   -f exps/customed_example/yolov_swinbase_v2.py\
#   -c /root/TemporalAttentionPlayground/YOLOV/V++_new_outputs/yolov_swinbase_v2/best_ckpt.pth\
#   -b 8 \
#   -d 1 \
#   --fp16 \
#   --lframe 0 \
#   --gframe 8 \

python tools/vid_eval.py \
  -f exps/customed_example/yolov_swinbase_v2.py\
  -c /root/TemporalAttentionPlayground/YOLOV/V++_new_outputs/yolov_swinbase_v2/best_ckpt.pth\
  -b 16 \
  -d 1 \
  --fp16 \
  --lframe 0 \
  --gframe 16 \

python tools/vid_eval.py \
  -f exps/customed_example/yolov_swinbase_v2.py\
  -c /root/TemporalAttentionPlayground/YOLOV/V++_new_outputs/yolov_swinbase_v2/best_ckpt.pth\
  -b 32 \
  -d 1 \
  --fp16 \
  --lframe 0 \
  --gframe 32 \



echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
