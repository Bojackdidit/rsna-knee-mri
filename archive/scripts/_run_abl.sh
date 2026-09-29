#!/bin/bash
# One variable at a time, as the 11th-place advice prescribes. Usage: _run_abl.sh ARCH RES POOL K TAG
#
# resnet34 @224 + mean pooling read 0.8565 and effnet-b0 read 0.8774, against our CoAtNet-384 +
# attention at 0.9133. Two candidate explanations and they need separating:
#   (a) the HEAD - mean-pooling 12 windows is noisy where per-finding attention can select. Run the
#       SAME small backbone with OUR attention head: if it jumps, the head is carrying the pipeline.
#   (b) the WINDOW COUNT - simple pooling may simply need far more samples to average over. Cheap to
#       test precisely because the model is small: k=48 on resnet34 costs less than k=12 on CoAtNet.
set -u
ARCH="$1"; RES="$2"; POOL="$3"; K="$4"; TAG="$5"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 12 \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch "$ARCH" --res "$RES" --pool "$POOL" \
  --k "$K" --k_eval "$K" --epochs 24 --bs 16 --bb_lr 3e-4 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
