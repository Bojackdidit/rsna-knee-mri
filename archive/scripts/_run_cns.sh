#!/bin/bash
# ConvNeXt-SMALL @384 - aiming at the gap the board reads leave open.
#
# Measured per-arm model time (total minus the fixed 3.7 min of reading; arms run CONCURRENTLY one
# per GPU, so this IS the wall-clock contribution):
#     CoAtNet-Base  20.1 min -> board 0.941
#     ConvNeXt-Base 15.4 min -> board 0.923
#     effv2s@288    12.2 min -> board 0.913
# The bar is under 15.9 min AND above 0.913, i.e. model under 12.2 min while scoring above 0.913.
# ConvNeXt-Base misses on time (19.1 total); effv2s@288 misses on score. ConvNeXt-Small sits between
# them in capacity (~50M vs ~88M params, roughly 0.55x the cost) and is the same family that already
# delivered 0.923 - so it is the best-motivated shot at clearing both.
#
# Trained WITHOUT the grouped split: this is a SHIPPING arm, so it sees everything but the 58 gold
# studies. The grouped gate has been shown not to rank for the board, so holding out 631 more studies
# would only weaken the arm for no information.
set -u
LABELS="${1:?labels parquet}"; TAG="${2:?tag}"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --arch convnext_small.fb_in22k_ft_in1k_384 --res 384 \
  --labels "$LABELS" \
  --k 6 --k_eval 12 --epochs 14 --bs 4 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --tag "$TAG"
echo "===== CNS DONE $TAG $(date -u +%H:%M:%S) ====="
