#!/bin/bash
# THE LABEL SWEEP. Same architecture, same schedule, same everything - only the labels change.
#
# I claimed the gap to Scott was labels. That was a guess by elimination, not a measurement, and it
# contradicts our own numbers: the one label comparison we have is +0.0075 (cd_ext -> blend, sd
# 0.0033), which cannot explain a 0.06 gap. So measure the axis instead of asserting it.
#
# Four of our six label sets have NEVER been scored on this gate, including labels_opus - the
# hand-read set. The two that have:
#     cd_ext      0.8851   (CoAtNet-224, 24 epochs, common cd_ext targets)
#     opus_blend  0.8883
#
# Settings are the RECORDED BASELINE's, so these are directly comparable to the 0.8821 reference:
# CoAtNet-224 @224, k=6/k_eval=12, 12 epochs, bs 4, bb_lr 3e-5, grad_ckpt, grouped fold 0.
#
# ⚠ EVERY arm here trains on a DIFFERENT label set, and this gate grades each arm against its own
# targets. The printed number is therefore NOT comparable across runs. The npz dumps are what matter:
# re-score them all against ONE common target set afterwards. That correction was worth 0.015-0.018
# the last time it was skipped.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--ckpt timm --norm imagenet --workers 6 --grad_ckpt \
 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --k 6 --k_eval 12 --epochs 12 --bs 4 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42"

run () {  # tag labelfile
  echo "===== LABELS $1 | $2 | $(date -u +%H:%M:%S) ====="
  python -u finetune_box.py $COMMON --labels "/workspace/knee/$2" --tag "$1"
}

run lab_opus    labels_opus.parquet
run lab_publab  labels_publab.parquet
run lab_pubrank labels_pubrank.parquet
run lab_cdrsna  labels_cd_rsna.parquet
echo "===== LABEL SWEEP DONE $(date -u +%H:%M:%S) ====="
