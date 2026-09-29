#!/bin/bash
# One-shot box setup: deps, BOTH corpus halves, merge, verify.
#
# The corpus ships as TWO Kaggle datasets - `knee-raptor-corpus` is titled "part 1 of 2" (3200
# studies) and `knee-raptor-corpus-ext` holds the other 1207. Pulling only part 1 is what crippled
# every arm trained earlier today: 2,524 training studies against the 4,349 the shipped arms had.
# This script always pulls both and refuses to hand over a corpus that is not 4407 studies.
set -eu

# ============================================================================
# 2026-09-13 GEOMETRY GATE - this script builds the OFF-GEOMETRY 44-slice corpus.
# ============================================================================
# Every serving kernel builds 96 slices (SLOTS 27/21/18/12/18, span 0.02-0.98). This script
# builds 44 (12/10/8/6/8, span 0.06-0.94). A window is three physically ADJACENT CORPUS slices
# as RGB, so at 44 that triplet spans 2.18x MORE anatomy than the triplet served at test time.
# Across twelve board reads and eight architectures, no arm trained on this corpus ever cleared
# 0.928, while the two v11-matched arms give 0.941.
# Use _setup_box_v11.sh instead. The v11 corpus is already built (Kaggle shard kernels
# knee-mri-fine-96-corpus-shard-0..4, 4407 studies, 0 failures) - it only needs assembling.
if [ "${ALLOW_44_SLICE_CORPUS:-0}" != "1" ]; then
  echo "*** REFUSING: this builds the 44-slice corpus and the kernel serves 96."
  echo "    Use:  bash _setup_box_v11.sh"
  echo "    Override only for a written-down off-geometry experiment:"
  echo "      ALLOW_44_SLICE_CORPUS=1 bash _setup_box.sh"
  exit 1
fi
echo "[geometry] WARNING: building the 44-slice corpus by explicit override; arms will not ship."

cd /workspace
chmod 600 /root/.kaggle/* 2>/dev/null || true
pip -q install timm pandas pyarrow scikit-learn kaggle 2>&1 | tail -1

mkdir -p /workspace/knee/corpus /workspace/ext
( cd /workspace/knee/corpus && kaggle datasets download -d dreaddevelopment/knee-raptor-corpus --unzip --force ) &
P1=$!
( cd /workspace/ext && kaggle datasets download -d dreaddevelopment/knee-raptor-corpus-ext --unzip --force ) &
P2=$!
wait $P1 $P2
echo "both halves downloaded"
ls -la /workspace/knee/corpus/*.npy /workspace/ext/*.npy | awk '{print $5,$9}'

python -u /workspace/_merge_corpus.py

python - <<'PY'
import numpy as np
v = np.load("/workspace/knee/corpus_full/all_vols.npy", mmap_mode="r")
i = np.load("/workspace/knee/corpus_full/all_ids.npy", allow_pickle=True)
assert v.shape == (4407, 44, 336, 336), "corpus is %r, expected 4407 studies" % (v.shape,)
assert len(i) == 4407 and len(set(i.astype(str))) == 4407
print("CORPUS OK", v.shape, "| 4407 unique ids")
PY
echo "BOX READY"
