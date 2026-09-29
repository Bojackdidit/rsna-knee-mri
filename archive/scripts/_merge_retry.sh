#!/bin/bash
# The Kaggle CDN drops mid-chunk on a 2 GB pull often enough that a single pass rarely finishes.
# Two breaks already killed it at chunk 24 of 25 and again at chunk 14. Retry until the merge's own
# verification passes; it re-places rows idempotently, so a repeat is wasted bandwidth, never a
# corrupt corpus.
cd /workspace/knee
for i in $(seq 1 12); do
  echo "=== attempt $i $(date -u +%H:%M:%S) ==="
  python -u _merge_v11.py --out /workspace/knee/corpus_v11 && { echo "MERGE COMPLETE on attempt $i"; break; }
  echo "--- attempt $i failed, retrying ---"
  sleep 10
done
python - <<'PYEOF'
import numpy as np, os
d="/workspace/knee/corpus_v11"
try:
    v=np.load(os.path.join(d,"all_vols.npy"), mmap_mode="r")
    m=np.load(os.path.join(d,"all_masks.npy"))
    i=np.load(os.path.join(d,"all_ids.npy"), allow_pickle=True)
    print("FINAL vols %s masks %s ids %d unique %d" % (v.shape, m.shape, len(i), len(set(i.astype(str)))))
    nz=int((v[::200].reshape(len(v[::200]),-1).max(1)>0).sum())
    print("FINAL sampled non-empty studies: %d of %d" % (nz, len(v[::200])))
except Exception as e:
    print("FINAL verification failed: %s" % e)
PYEOF
