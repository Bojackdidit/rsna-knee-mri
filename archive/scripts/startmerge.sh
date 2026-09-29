#!/bin/bash
# ONE clean v11 corpus merge, with exponential backoff.
#
# History of why this file exists, so it is not repeated:
#  * Two _merge_v11.py processes ran concurrently on the same output memmap. They raced on the
#    Kaggle API (every call returned exit 1 instantly) and on the staging chunks (one deleted a
#    .npy while the other was loading it -> "EOFError: No data left in file").
#  * The first retry wrapper burned all 12 of its attempts in 2 minutes because of that.
#  * `pkill -f "_merge_retry"` matched the ssh command line carrying that same string and killed
#    the session before the cleanup ran. Hence: a FILE, never an inline pattern.
#
# ALWAYS check `pgrep -af "[_]merge_v11" | wc -l` is 0 before starting this.
cd /workspace/knee
rm -rf corpus_v11/_stage corpus_v11/all_vols.npy corpus_v11/all_masks.npy corpus_v11/all_ids.npy
mkdir -p corpus_v11
for i in $(seq 1 20); do
  echo "=== attempt $i $(date -u +%H:%M:%S) ==="
  if python -u _merge_v11.py --out /workspace/knee/corpus_v11; then
    echo "MERGE COMPLETE attempt $i"
    break
  fi
  b=$(( i * 30 )); [ $b -gt 300 ] && b=300
  echo "--- failed, sleeping ${b}s ---"
  sleep $b
done
python - <<'PYEOF'
import numpy as np, os
d = "/workspace/knee/corpus_v11"
v = np.load(os.path.join(d, "all_vols.npy"), mmap_mode="r")
m = np.load(os.path.join(d, "all_masks.npy"))
ids = np.load(os.path.join(d, "all_ids.npy"), allow_pickle=True)
print("FINAL vols %s masks %s ids %d unique %d" % (v.shape, m.shape, len(ids), len(set(ids.astype(str)))))
val = m.mean(0); acc = 0
for s, b in enumerate([27, 21, 18, 12, 18]):
    print("  slot %d validity %.3f" % (s, val[acc:acc + b].mean())); acc += b
samp = v[::200]
nz = (samp.reshape(len(samp), -1).sum(1) > 0).mean()
print("NONZERO fraction %.3f  <-- a full-size npy can still be all zeros; this is the real check" % nz)
PYEOF
echo "STARTMERGE DONE $(date -u +%H:%M:%S)"
