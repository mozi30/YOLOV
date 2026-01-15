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

perturbations=(
  gaussian_noise
  motion_blur
  jpeg_compression
  brightness_change
  contrast_change
  pixelation
  defocus_blur
)

perturbations2=(
  defocus_blur
)

severities=(low med high)

# for p in "${perturbations[@]}"; do
#   for s in "${severities[@]}"; do
#     echo "Evaluating with perturbation: $p, severity: $s"

#     python tools/vid_eval.py \
#       -f exps/customed_example/yolov_swinbase_v2.py\
#       -c $HOME_DIR/weights/yolov/yolov.pth\
#       -b 2 \
#       -d 1 \
#       --fp16 \
#       --lframe 0 \
#       --gframe 2 \
#       --stride 10 \
#       --perturbation \
#       --select_perturbation $p \
#       --severity $s

#       echo "------------------------------------------"
#   done
# done



# for p in "${perturbations[@]}"; do
#   for s in "${severities[@]}"; do
#     echo "Evaluating with perturbation: $p, severity: $s"

#     python tools/vid_eval.py \
#       -f exps/customed_example/yolov_swinbase_v2.py\
#       -c $HOME_DIR/weights/yolov/yolov.pth\
#       -b 4 \
#       -d 1 \
#       --fp16 \
#       --lframe 0 \
#       --gframe 4 \
#       --stride 10 \
#       --perturbation \
#       --select_perturbation $p \
#       --severity $s

#       echo "------------------------------------------"
#   done
# done

# for p in "${perturbations[@]}"; do
#   for s in "${severities[@]}"; do
#     echo "Evaluating with perturbation: $p, severity: $s"

#     python tools/vid_eval.py \
#       -f exps/customed_example/yolov_swinbase_v2.py\
#       -c $HOME_DIR/weights/yolov/yolov.pth \
#       -b 8 \
#       -d 1 \
#       --fp16 \
#       --lframe 0 \
#       --gframe 8 \
#       --stride 10 \
#       --perturbation \
#       --select_perturbation $p \
#       --severity $s

#       echo "------------------------------------------"
#   done
# done

python tools/vid_eval.py \
      -f exps/customed_example/yolov_swinbase_v2.py\
      -c $HOME_DIR/weights/yolov/yolov.pth\
      -b 2 \
      -d 1 \
      --fp16 \
      --lframe 0 \
      --gframe 2 \
      --stride 10 

python tools/vid_eval.py \
      -f exps/customed_example/yolov_swinbase_v2.py\
      -c $HOME_DIR/weights/yolov/yolov.pth\
      -b 4 \
      -d 1 \
      --fp16 \
      --lframe 0 \
      --gframe 4 \
      --stride 10 

python tools/vid_eval.py \
      -f exps/customed_example/yolov_swinbase_v2.py\
      -c $HOME_DIR/weights/yolov/yolov.pth\
      -b 8 \
      -d 1 \
      --fp16 \
      --lframe 0 \
      --gframe 8 \
      --stride 10 


echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
