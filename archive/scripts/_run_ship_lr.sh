#!/bin/bash
# SHIPPING arm at the LR the curve actually chose. Usage: _run_ship_lr.sh BBLR LABELS SEED TAG
#
# This is the arm the 4-point LR curve exists to specify. The shipped champion pair has a backbone
# that moved 0.0220 from ImageNet init against a head that moved 0.8953 - i.e. it was never trained.
# 3e-4 was measured to DIVERGE on this backbone (0.6948 -> 0.5691, loss flat at 0.94), so the usable
# band is at or below 1e-4 and the curve (3e-5 / 5e-5 / 1e-4) picks the point.
#
# FULL corpus, NO grouped holdout: the trainer already excludes gold-58, so this trains on all 4,349
# studies exactly as the champion arms did. The grouped gate is deliberately given up here because it
# does not rank in board order - holding out 870 studies would cost real training data for a number
# that has already been shown not to predict.
#
# k = k_eval = 12 so the arm is trained AT the budget it ships at: the mismatch on the champion
# (trained k=6, inferred K=48) costs 0.0123 when the window budget moves, and matched-k training cuts
# that to 0.0008. Interleaved at serve time, the pair still covers 24 positions for 12 of work.
#
# 24 epochs, not 16: b0 gained +0.0116 going 12 -> 24, and a higher backbone LR needs MORE schedule,
# not less, because OneCycle spends the early epochs ramping.
set -u
BBLR="$1"; LABELS="$2"; SEED="$3"; TAG="$4"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --k 12 --k_eval 12 --epochs 24 --bs 2 --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 \
  --labels "$LABELS" --seed "$SEED" --tag "$TAG"
echo "===== $TAG DONE $(date -u +%H:%M:%S) | $(grep -aoE 'best 0\.[0-9]+' arm_$TAG.log 2>/dev/null | tail -1) ====="
