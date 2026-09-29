#!/bin/bash
# Does the CoAtNet backbone actually train? Usage: _run_lr.sh BBLR TAG
#
# Weight-space diff of the two SHIPPED arms: backbone moved 0.0220 from ImageNet init, head moved
# 0.8953. At bb_lr 3e-5 we ship an attention head on FROZEN ImageNet features. Measured on another
# family: efficientnet-b0 read 0.8158 at 3e-5 and 0.8598 at 3e-4 - +0.044 from the LR alone.
#
# The only prior CoAtNet test at a higher LR is INVALID: it ran on the part-1-only corpus (3,155 of
# 4,349 studies) and was still rising when it stopped. So the champion family has never been trained
# at its correct LR on the full corpus. This settles it.
#
# Grouped fold 0 held out and the SAME labels in every run, so this is the one comparison the gate is
# actually valid for: same arm, same labels, one variable. Layer-wise LR decay so the early blocks
# move less than the late ones.
set -u
BBLR="$1"; TAG="$2"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --k 12 --k_eval 12 --epochs 16 --bs 2 --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
echo "===== $TAG DONE $(date -u +%H:%M:%S) | $(grep -aoE 'best 0\.[0-9]+' arm_$TAG.log 2>/dev/null | tail -1) ====="
