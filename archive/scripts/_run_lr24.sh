#!/bin/bash
# Same comparison, LONGER schedule. Usage: _run_lr24.sh BBLR TAG
#
# The 16-epoch test has a confound: OneCycleLR spends the early epochs ramping toward max_lr, so a
# higher backbone LR is still recovering from disturbing the ImageNet features while a low one has
# been converging the whole time. efficientnet-b0 gained +0.0116 going 12 -> 24 epochs, and a higher
# LR needs MORE schedule, not less. If 3e-5 wins at 16 epochs that could mean 3e-5 is right, or it
# could mean 16 epochs is too short to let 1e-4 finish annealing - and those have opposite
# consequences for what we ship.
#
# Identical to _run_lr.sh in every other respect (CoAtNet-384, k=k_eval=12, grouped fold 0, blend
# labels, seed 42) so it stays comparable both to the 16-epoch runs and to its own LR sibling.
set -u
BBLR="$1"; TAG="$2"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --k 12 --k_eval 12 --epochs 24 --bs 2 --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
echo "===== $TAG DONE $(date -u +%H:%M:%S) =====" >> /workspace/chain24.log
