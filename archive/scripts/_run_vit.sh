#!/bin/bash
# A SECOND ViT-B/16 @384 arm, so a ViT PAIR becomes shippable.
#
# Arms run one per GPU IN PARALLEL, so wall-clock is the SLOWER arm, not the sum. A pair should
# therefore be two arms of SIMILAR cost - pairing a cheap arm with an expensive one buys nothing.
# Measured per-study cost at 384/48 windows:
#     CoAtNet 88.3ms (1.00x) | ConvNeXt-Base 47.6 (0.54x) | ViT-B/16 22.6 (0.26x)
# The champion's model term is 20.1 of its 23.8 min, so a ViT pair projects to ~5.2 min of model plus
# 3.7 reading = ~8.9 min. We already hold ONE ViT@384 arm (confdist labels, gold 0.9087); this is its
# blend-label partner, matching how the shipped CoAtNet pair is built: same arch, two label sets.
#
# NO --group_csv. This arm is for SHIPPING, not the gate, so it trains on everything except the 58
# gold studies. The grouped split holds out 631 more and would make a weaker arm - and the gate has
# just been shown not to rank for the board anyway.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --arch vit_base_patch16_384 --res 384 \
  --labels /workspace/knee/labels_opus_blend.parquet \
  --k 6 --k_eval 12 --epochs 14 --bs 4 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --tag vit384_blend
echo "===== VIT ARM DONE $(date -u +%H:%M:%S) ====="
