#!/bin/bash
# The spec build. Usage: _run_spec.sh ARCH RES SLOTS K EPOCHS EMA MIXUP DROPPATH FOLD BS TAG
#
# EfficientNetV2-S @288, 3 series (corpus slots 0,2,4), k_train = k_eval = 12 per lane interleaved
# to 24 distinct positions, 48 epochs with EMA 0.9998 + mixup 0.2 + drop_path 0.2, bb_lr 3e-4
# (the CONVNET value - it diverges on hybrids), full augmentation, scanner-grouped folds.
#
# Grounded: EfficientNet beat ResNet on every matched pair we ran; effv2s-288 x2 scored 0.913 on the
# board while carrying three handicaps since removed (58% of the corpus, 12 epochs, no augmentation).
# 3 series measured ~= 5 series offline (0.9211 vs 0.9224) while 2 series is board-refuted at -0.038.
# K below 12 is a 6x net loss on the efficiency metric (0.001 AUC = 72 s; K 12->9 costs 0.0043).
#
# Two bugs were fixed before this could run at all:
#   --slots indexed CORPUS_SLOTS (the 2-slot 224 subset, 2 entries) so `--slots 0,2,4` raised
#   IndexError; and window centres were drawn contiguously between the first and last valid slice,
#   so ~19% of all windows ever trained straddled a slot boundary and stacked two DIFFERENT series
#   into one RGB triple. On slots 0,2,4 the old code produced 42 centres of which 20 were invalid.
set -u
ARCH="$1"; RES="$2"; SLOTS="$3"; K="$4"; EPOCHS="$5"; EMA="$6"; MIXUP="$7"; DP="$8"; FOLD="$9"; BS="${10}"; TAG="${11}"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# RECLAIM=0.7 gives most of the held-out fold BACK to training and validates on what is left.
# Shipping arms should use it: holding out a whole fifth of the data to measure a number costs real
# accuracy in the model that actually gets sent. Measurement runs leave it off so the gate stays
# clean. Dread asked for this explicitly - fine-tune the withheld studies back in, about 70%.
# BB_LR must be set per FAMILY. 3e-4 is the convnet value and is right for EfficientNet/ResNet;
# on CoAtNet-384 it DIVERGES (measured: loss flat at 0.94, killed at ep3). CoAtNet wants 3e-5,
# which led at every matched epoch against 5e-5, 1e-4 and 3e-4. Leaving this hardcoded would burn
# a whole run.
# A PAIR gets its strength from disagreement, not from two copies of the same model. The champion
# pair is two arms on two DIFFERENT label sets, and cross-label pairs have consistently blended
# better here than same-label ones. LABELS is the lever for the second arm.
LABELS="${LABELS:-/workspace/knee/labels_opus_blend.parquet}"
BB_LR="${BB_LR:-3e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
SEED="${SEED:-42}"
# Gradient checkpointing: needed for CoAtNet-384 at any useful batch size.
CKPT_ARG=""
[ "${GRAD_CKPT:-0}" != "0" ] && CKPT_ARG="--grad_ckpt"
# LLRD=0.75 trains the top of the backbone at BB_LR and decays toward the input. The flat-rate
# sweeps all failed (1e-4 -0.014, 3e-4 diverged) because one rate hit all 72.9M params at once.
LLRD_ARG=""
[ -n "${LLRD:-}" ] && LLRD_ARG="--llrd $LLRD --llrd_groups ${LLRD_GROUPS:-8}"
RECLAIM="${RECLAIM:-0}"
RECLAIM_ARG=""
[ "$RECLAIM" != "0" ] && RECLAIM_ARG="--reclaim $RECLAIM"
EXTRA=""
[ -n "$SLOTS" ] && EXTRA="--slots $SLOTS"
python -u finetune_box.py --ckpt timm --norm imagenet --workers 16 \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold "$FOLD" ${RECLAIM_ARG} \
  --arch "$ARCH" --res "$RES" --pool "${POOL:-attn}" --aug geo,erase,blur,noise $EXTRA \
  --k "$K" --k_eval "$K" --epochs "$EPOCHS" --bs "$BS" \
  --ema "$EMA" --mixup "$MIXUP" --drop_path "$DP" \
  --bb_lr "${BB_LR:-3e-4}" --head_lr "${HEAD_LR:-1e-3}" --wd 0.02 --seed "${SEED:-42}" ${CKPT_ARG} ${LLRD_ARG} \
  --labels "${LABELS:-/workspace/knee/labels_opus_blend.parquet}" --tag "$TAG"
