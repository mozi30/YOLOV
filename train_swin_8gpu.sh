#!/bin/bash
# YOLOX-Swin-Base Training Script for 8x GPU
# Optimized for VisDrone dataset

set -e

cd /root/TemporalAttentionPlayground/YOLOV
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolox

echo "=========================================="
echo "YOLOX-Swin-Base Training (8x GPU)"
echo "=========================================="
echo "Dataset: VisDrone"
echo "GPUs: 8x RTX 3090"
echo "Backbone: Swin-Base Transformer"
echo "Input Size: 608×1088"
echo "Batch Size: 64 (8 per GPU)"
echo "Epochs: 80"
echo "Expected Time: ~19-20 hours"
echo "=========================================="
echo ""

# Check if pretrained weights exist
if [ ! -f "pretrained/swin_base_patch4_window7_224_22k.pth" ]; then
    echo "ERROR: Swin-Base pretrained weights not found!"
    echo "Please download from:"
    echo "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
    echo "And save to: pretrained/swin_base_patch4_window7_224_22k.pth"
    exit 1
fi

echo "Starting training..."
echo ""

# Option 1: Standard resolution (608×1088) - Recommended
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    -f exps/yolov_ovis/yolox-swin-visdrone-8gpu.py \
    -d 8 \
    -b 64 \
    --fp16 \
    -c pretrained/swin_base_patch4_window7_224_22k.pth

# Option 2: High resolution (768×1344) - Uncomment for best accuracy
# echo "Training with HIGH RESOLUTION (768×1344)..."
# echo "Note: Reduce batch to 32 (4 per GPU) for memory"
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     --master_port=29500 \
#     tools/train.py \
#     -f exps/yolov_ovis/yolox-swin-visdrone-8gpu.py \
#     -d 8 \
#     -b 32 \
#     --fp16 \
#     -c pretrained/swin_base_patch4_window7_224_22k.pth

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox-swin-visdrone-8gpu/"
echo "=========================================="
