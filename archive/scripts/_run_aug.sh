#!/bin/bash
# Augmentation ablation. Usage: _run_aug.sh ARCH RES POOL K AUG BBLR TAG
set -u
ARCH="$1"; RES="$2"; POOL="$3"; K="$4"; AUG="$5"; BBLR="$6"; TAG="$7"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 12 --grad_ckpt \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch "$ARCH" --res "$RES" --pool "$POOL" --aug "$AUG" \
  --k "$K" --k_eval "$K" --epochs 24 --bs 16 --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
