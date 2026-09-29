#!/bin/bash
# The shipping arms, retrained with 70% of the held-out fold returned to training.
# Usage: _run_reclaim.sh ARCH BBLR BS SEED TAG
#
# Dread's instruction: "what I want is for that set of with-held data to be fine tuned into the model,
# not the entire set not every single one, I am thinking 70% of the 800 is fine".
#
# Why it should pay, measured rather than assumed: the shipped champion arms train on all 4,349
# studies with no holdout, ours surrender 870 to have a gate, and at MATCHED window count the
# champion pair scored 0.931 against our 0.922 - a 0.009 gap with the data split as the only
# structural difference left between them.
#
# The 30% that stays back is still a real validation set: the split is by scanner FINGERPRINT, so no
# scanner appears on both sides. Splitting by study instead would hand the validation set its own
# training data and every number after it would be inflated.
#
# Everything else is byte-identical to the recipes that produced the two arms in the 0.922 pair:
#   ref24            coatnet_rmlp_2_rw_384  bb_lr 3e-5  bs 2 + grad_ckpt  seed 42   gate 0.9165
#   effv2s_384_24b   tf_efficientnetv2_s    bb_lr 3e-4  bs 4              seed 777  gate 0.9123
set -u
ARCH="$1"; BBLR="$2"; BS="$3"; SEED="$4"; TAG="$5"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 12 --grad_ckpt \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 --reclaim 0.7 \
  --arch "$ARCH" --res 384 --k 12 --k_eval 12 \
  --epochs 24 --bs "$BS" --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 --seed "$SEED" \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
