#!/bin/bash
# STAGE 1 - is efficientnet a better BASE than CoAtNet, and does it need a real backbone LR?
#
# Scott: "I have really only ever spent time on efficientnet and resnet", single model, single fold,
# ~0.949 and the fastest run on the board. We run two CoAtNet-384 arms and score 0.941 in 23.8 min.
#
# The confound this removes: bb_lr 3e-5 was tuned for a CoAtNet carrying Raptor SSL weights, and it
# is on record that at that LR the backbone barely moves (0.0220 drift against 0.8953 in the head) -
# our arms are near frozen-backbone + trained head. An ImageNet efficientnet_b0 needs its backbone
# actually trained, so sweep the backbone LR before judging the family. Otherwise a null is
# unreadable: "efficientnet is worse" and "efficientnet was never trained" look identical.
#
# bs 4 and grad_ckpt-free: the CoAtNet baseline was FORCED to bs 4 by a 24GB card, and matching it
# exactly is what makes 0.8821 a comparable number. b0 does not need grad_ckpt (memory only, no
# effect on the result), so it is dropped. Batch size is separately measured as a minor effect
# (bs 16 -> 0.8996 vs bs 8 -> 0.9032), and the direction favours the SMALLER batch anyway.
#
# Run CONCURRENTLY: bs 4 at 224 with k=6 is 24 images per step, a fraction of a 24GB 3090, and
# the box has 96 CPUs for the loaders. Three at once costs far less wall-clock than three in series.
# Gate: grouped fold 0, 631 scanner-disjoint studies, 12 epochs. CoAtNet baseline on it = 0.8821.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--ckpt timm --norm imagenet --workers 6 \
 --labels /workspace/knee/labels_cd_ext.parquet \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --k 6 --k_eval 12 --epochs 12 --head_lr 1e-3 --wd 0.02 --seed 42 \
 --arch efficientnet_b0 --res 224 --bs 4"

for LR in 3e-5 3e-4 1e-3; do
  TAG="b0_lr${LR}"
  echo "launch $TAG at $(date -u +%H:%M:%S)"
  nohup python -u finetune_box.py $COMMON --bb_lr $LR --tag "$TAG" > "arm_${TAG}.log" 2>&1 &
  sleep 20
done
wait
echo "===== STAGE 1 DONE $(date -u +%H:%M:%S) ====="
for f in arm_b0_lr*.log; do
  echo "--- $f"; grep -aE "^ep |BEST|macro|Error|Traceback|OutOfMemory" "$f" | tail -6
done
