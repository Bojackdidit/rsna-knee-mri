#!/bin/bash
# One-shot box setup at SERVE geometry. Replaces _setup_box.sh.
#
# WHY THIS EXISTS
# ---------------
# _setup_box.sh pulls the 44-slice v2 corpus and asserts (4407, 44, 336, 336). Every serving
# kernel builds 96 slices (SLOTS 27/21/18/12/18, span 0.02-0.98). A window is three physically
# ADJACENT CORPUS slices stacked as RGB, so at 44 that triplet spans 2.18x MORE anatomy than the
# triplet the kernel hands the model at test time - the third channel, the only 3D information
# the model has, is systematically wrong at inference. Across twelve board reads and eight
# architectures, no box-trained arm ever cleared 0.928 while the two v11-matched arms give 0.941.
#
# The v11 corpus does NOT need rebuilding. It was built 2026-08-26 and is complete: 5 shards x 5
# chunks, 4,407 studies, 0 failures. This pulls those kernel outputs from the Kaggle CDN straight
# to the box - no home bandwidth, no new dataset to create, no 20 GB /kaggle/working cap.
#
# Needs ~50 GB free. Takes roughly 10-20 min on a typical Vast link.
set -eu
cd /workspace
chmod 600 /root/.kaggle/* 2>/dev/null || true
pip -q install timm pandas pyarrow scikit-learn kaggle 2>&1 | tail -1

OUT=/workspace/knee/corpus_v11
mkdir -p "$OUT"

# Refuse to start a 48 GB assembly that cannot finish.
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 60 ]; then
  echo "*** ABORT: only ${FREE_GB} GB free on /workspace; the v11 corpus needs ~50 GB plus headroom."
  echo "    Destroy and re-rent with a bigger disk rather than half-filling it."
  exit 1
fi
echo "[disk] ${FREE_GB} GB free - proceeding"

# Pull + concatenate + verify. Chunks are deleted as they land, so peak disk stays ~50 GB.
python -u /workspace/knee/_merge_v11.py --out "$OUT"

# Independent re-check against what the KERNEL serves. _merge_v11.py already asserts this;
# doing it again here means a hand-edited merge can never quietly hand over the wrong geometry.
python - <<'PY'
import numpy as np
SERVE_MAXS = 96                       # MAXS in every kernel under variants/
SLOTS = [27, 21, 18, 12, 18]
v = np.load("/workspace/knee/corpus_v11/all_vols.npy", mmap_mode="r")
m = np.load("/workspace/knee/corpus_v11/all_masks.npy")
i = np.load("/workspace/knee/corpus_v11/all_ids.npy", allow_pickle=True)
assert v.shape == (4407, SERVE_MAXS, 336, 336), "corpus is %r, expected 96-slice v11" % (v.shape,)
assert sum(SLOTS) == SERVE_MAXS, "slot budgets do not sum to MAXS"
assert len(i) == 4407 and len(set(i.astype(str))) == 4407
val = m.mean(0); acc = 0
for s, b in enumerate(SLOTS):
    print("  slot %d slices %2d-%2d validity %.3f" % (s, acc, acc + b - 1, val[acc:acc + b].mean()))
    acc += b
print("CORPUS OK", v.shape, "| 4407 unique ids | matches kernel MAXS =", SERVE_MAXS)
PY

echo
echo "BOX READY (v11 / 96-slice / serve geometry)"
echo "  export KNEE_DATA_ROOT=$OUT"
echo "  finetune_box.py will pass its geometry gate; it REFUSES a 44-slice corpus now."
