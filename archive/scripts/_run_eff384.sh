#!/bin/bash
# effv2s @384, k = k_eval = 12. The config that can actually clear Dread's bar. Usage: _run_eff384.sh SEED TAG
#
# Everything trained tonight was at 224 to save GPU time, and that capped the ceiling. The board has
# already priced resolution: a 224 CoAtNet pair scored 0.928, a 384 pair scores 0.941. So no 224 pair
# reaches 0.941 whatever the family - the gate agrees, with every 224 pair landing 0.88-0.90.
#
# 384 is required for a 0.941-class score, and effv2s still fits the clock there:
#     effv2s@288 measured 12.2 min of model at K=48 on the board
#     at 384 that is 12.2 * (384/288)^0.6 = 14.5 min at K=48, so 3.6 min at K=12
#     plus ~3.0 min of slice-limited read  ->  ~6.6 min total, inside the 8-minute bar
#
# k = k_eval = 12 because training AT the inference budget removes the window-count penalty: on
# CoAtNet, k=12 (0.9071) and k=6 (0.9063) came out 0.0008 apart, while the shipped champion - trained
# at k=6 and inferred at 48 - lost 0.0123 when cut to K=6.
#
# Blend labels on both arms, decorrelated by seed: on COMMON targets cd_ext is 0.0039 weaker for this
# family, so pairing by label set costs more than it buys.
set -u
SEED="$1"; TAG="$2"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch tf_efficientnetv2_s.in21k_ft_in1k --res 384 \
  --k 12 --k_eval 12 --epochs 18 --bs 4 --bb_lr 3e-4 --head_lr 1e-3 --wd 0.02 \
  --labels /workspace/knee/labels_opus_blend.parquet --seed "$SEED" --tag "$TAG"
echo "===== $TAG DONE $(date -u +%H:%M:%S) | $(grep -aoE 'best 0\.[0-9]+' arm_$TAG.log 2>/dev/null | tail -1) ====="
