#!/bin/bash
# Lever 3: does telling the head WHICH SEQUENCE each window came from help? Usage: _run_slot.sh TAG
#
# Byte-for-byte the control run (CoAtNet-384, k=k_eval=12, grouped fold 0, blend labels, seed 42,
# 16 epochs, bb_lr 3e-5 - the LR the curve just showed is right) with ONE addition: --slot_emb. So it
# is directly comparable to coat384_ref = 0.9133 and nothing else differs.
#
# Why it should matter: every finding currently reads the same globally-pooled per-window vector and
# the head cannot tell a sagittal window from a coronal one, so per-finding window attention cannot
# prefer the plane a finding is visible in. The measured symptom is outputs that over-correlate
# (+0.168 median over all 66 pairs) with the weakest findings being the localized ones.
#
# The embedding is ZERO-initialised, so this starts bit-identical to the control (verified offline:
# max|diff| = 0.000e+00) and can only diverge where the optimiser finds it pays. 5 x F_dim params.
set -u
TAG="$1"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt --slot_emb \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --k 12 --k_eval 12 --epochs 16 --bs 2 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
