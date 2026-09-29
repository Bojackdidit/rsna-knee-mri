#!/bin/bash
# SHIPPING arms: CoAtNet-384 trained at k = k_eval = 24. Usage: _run_ship24.sh SEED LABELS TAG
#
# Dread's absolutes: beat 0.941 AND beat 23 min. Equal is a failure. Every "faster" change measured
# tonight traded accuracy away (4.2x against, every time). This is the one change that pushes BOTH
# directions at once, and it comes straight out of a measurement:
#
#   the shipped champion is trained at k=6 windows and INFERRED at K_EVAL=48.
#   that mismatch costs 0.0123 when the window budget moves (0.9453 -> 0.9330 at K=6),
#   while an arm trained AT its budget loses almost nothing (k=12 0.9071 vs k=6 0.9063).
#
# So train at k=24 and ship interleaved at 24 per arm: arm 0 takes the even window positions and arm 1
# the odd, the pair still covers all 48 - confirmed on the board (0.940 with the mismatched champion
# arms) - but each arm now does HALF the work AND is no longer being asked to decide from a window
# count it never saw in training. Score should go up because the mismatch is gone; time should go down
# because K per arm halves.
#
# FULL 4,407-study corpus and NO grouped holdout - gold-58 is excluded by the trainer anyway. These are
# shipping arms, so they get all the data, exactly like the champion arms did. The grouped gate does
# not rank in board order, so holding out 870 studies to measure them would cost data for a number
# that has already been shown not to predict.
set -u
SEED="$1"; LABELS="$2"; TAG="$3"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --k 24 --k_eval 24 --epochs 14 --bs 2 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 \
  --labels "$LABELS" --seed "$SEED" --tag "$TAG"
echo "===== $TAG DONE $(date -u +%H:%M:%S) | $(grep -aoE 'best 0\.[0-9]+' arm_$TAG.log 2>/dev/null | tail -1) ====="
