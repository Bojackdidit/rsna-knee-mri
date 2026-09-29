#!/bin/bash
# Train the remaining scanner-grouped folds of a chosen recipe, for fold-plexing in the kernel.
# Usage: _run_folds.sh ARCH RES SLOTS K EPOCHS EMA MIXUP DP BS "FOLDLIST" PREFIX
#
# Fold 0 is already trained by the E-chain, so this normally runs "1 2 3". Each fold holds out a
# DIFFERENT scanner-disjoint fifth, so across the set every study is trained on by 4 of 5 arms -
# which is what Dread asked for when he said the held-out data should end up back in the model,
# without ever giving a single arm its own validation set to memorise.
#
# The kernel maps arm i to lane i%2 and rank-mean blends any even number of arms, so 2 folds per
# lane ships as a 4-arm plex. Model time per lane DOUBLES, so only plex into runtime headroom that
# has actually been measured on the board - never into a projected one.
set -u
ARCH="$1"; RES="$2"; SLOTS="$3"; K="$4"; EPOCHS="$5"; EMA="$6"; MIXUP="$7"; DP="$8"; BS="$9"
FOLDS="${10}"; PREFIX="${11}"
cd /workspace/knee
for f in $FOLDS; do
  TAG="${PREFIX}_f${f}"
  if [ -f "raptor_ft_${TAG}.pt" ]; then
    echo "===== $TAG already trained, skipping ====="
    continue
  fi
  echo "===== $TAG START $(date -u +%H:%M:%S) ====="
  bash _run_spec.sh "$ARCH" "$RES" "$SLOTS" "$K" "$EPOCHS" "$EMA" "$MIXUP" "$DP" "$f" "$BS" "$TAG" \
    > "run_${TAG}.log" 2>&1
  echo "===== $TAG DONE $(date -u +%H:%M:%S) | $(grep -aoE 'best 0\.[0-9]+' run_${TAG}.log 2>/dev/null | tail -1) ====="
done
