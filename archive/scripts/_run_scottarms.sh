#!/bin/bash
# SCOTT'S FAMILIES, built to the new standard. EfficientNet and ResNet - what he actually says he uses.
#
# These families were tested once already and lost, but not fairly: efficientnet_b0@224 0.8598,
# effv2s@288 0.8806, resnet50@224 0.8440 against CoAtNet-224's 0.8821 - all trained at k=6, inferred
# at a DIFFERENT window budget, and mostly on cd_ext labels. effv2s landed within 0.0015 of CoAtNet
# despite that. The recipe was the handicap, not the family.
#
# Three corrections, each one measured rather than guessed:
#   k = k_eval          trained AT the inference window budget. Every arm we ship was trained at k=6
#                       and inferred at 48, and that mismatch is why cutting windows costs accuracy.
#   bb_lr 3e-4          efficientnet's backbone barely trains at the inherited 3e-5 - measured worth
#                       +0.0440 on b0 (0.8158 -> 0.8598). Never carry a backbone LR across families.
#   blend labels        the only effect that cleared the paired-bootstrap noise floor (+0.0075,
#                       sd 0.0033); architecture and seed did not.
#
# And these are the families that reach the CLOCK. Measured per-arm model time at K=48 on the board:
# CoAtNet-384 20.1 min, effv2s@288 12.2. At 224 that is ~10.5, so K=6 is ~1.3 min of model; with
# slice-limited decode the read is ~1.5, i.e. **under three minutes total** - Scott's territory.
#
# Reference to beat on this gate: the champion CoAtNet arm at K=12 = 0.9375, at K=6 = 0.9330.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BLEND=/workspace/knee/labels_opus_blend.parquet
COMMON="--ckpt timm --norm imagenet --workers 6 \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --labels $BLEND --epochs 16 --bs 4 --bb_lr 3e-4 --head_lr 1e-3 --wd 0.02 --seed 42"

run () {   # tag arch res k
  echo "===== $1 | $2 @ $3 | k=k_eval=$4 | $(date -u +%H:%M:%S) ====="
  python -u finetune_box.py $COMMON --arch "$2" --res "$3" --k "$4" --k_eval "$4" --tag "$1"
}

# the 4-minute candidate first: Scott's family, low res, few windows, trained for exactly that
run eff_224_k6   tf_efficientnetv2_s.in21k_ft_in1k 224 6
run eff_224_k12  tf_efficientnetv2_s.in21k_ft_in1k 224 12
run rn50_224_k12 resnet50                          224 12
echo "===== SCOTT ARMS DONE $(date -u +%H:%M:%S) ====="
