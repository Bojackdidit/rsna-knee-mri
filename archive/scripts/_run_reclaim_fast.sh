#!/bin/bash
# Same as _run_reclaim.sh but sized for the card we are actually on. Usage: ARCH BBLR BS SEED TAG
# bs 2 + gradient checkpointing exists because every earlier box was a 24GB 4090. On a 95GB H100 that
# is pure waste - it ran ~450 s/epoch, which would not finish inside the remaining credit. bs 16 with
# checkpointing off uses the card. bs does change the recipe relative to ref24 (bs 2), so this arm is
# judged on the BOARD and against the 350-study gate, not by direct comparison to ref24's 0.9165.
set -u
ARCH="$1"; BBLR="$2"; BS="$3"; SEED="$4"; TAG="$5"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 12 \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 --reclaim 0.7 \
  --arch "$ARCH" --res 384 --k 12 --k_eval 12 \
  --epochs 24 --bs "$BS" --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 --seed "$SEED" \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
