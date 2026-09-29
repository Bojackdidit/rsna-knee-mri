#!/usr/bin/env python3
"""Ship only what the fast arm needs: two series, already at 224.

The corpus is (3200, 44, 336, 336) uint8 = 15.9 GB. The fast arm trains on corpus slots 0 and 2
only - Sagittal-FS (slices 0:12) and Coronal-FS (22:30) - which is 20 of the 44 slices, and it
trains at 224, so the 336 pixels are resized away on the first batch regardless. Doing both here
turns a 15.9 GB upload into 3.2 GB, which is the difference between a thirty-minute job and a
ninety-minute one.

The result is a contiguous 20-slice block, so on the box it is slots [(0,12),(12,20)] and the
training call becomes --slots '0,1'.
"""
import numpy as np, cv2, os, sys
SRC = r"C:/Users/johna/AppData/Local/Temp/claude/D--Artificial-Nervous-System/efa12c9d-7852-4b73-9032-6dc08d47204d/scratchpad/rsna_knee/raptor"
DST = sys.argv[1] if len(sys.argv) > 1 else r"D:/claude2/main_track/knee/corpus_s02"
RES = 224
os.makedirs(DST, exist_ok=True)
vols = np.load(os.path.join(SRC, "all_vols.npy"), mmap_mode="r")
ids  = np.load(os.path.join(SRC, "all_ids.npy"), allow_pickle=True)
mask = np.load(os.path.join(SRC, "all_masks.npy"), allow_pickle=True)
keep = list(range(0, 12)) + list(range(22, 30))
N, S = vols.shape[0], len(keep)
print("source", vols.shape, "-> subset (%d, %d, %d, %d)" % (N, S, RES, RES), flush=True)
out = np.lib.format.open_memmap(os.path.join(DST, "all_vols.npy"), mode="w+",
                                dtype=np.uint8, shape=(N, S, RES, RES))
for i in range(N):
    blk = vols[i][keep]
    for j in range(S):
        out[i, j] = cv2.resize(blk[j], (RES, RES), interpolation=cv2.INTER_AREA)
    if i % 400 == 0: print("  %d/%d" % (i, N), flush=True)
out.flush()
np.save(os.path.join(DST, "all_ids.npy"), ids)
np.save(os.path.join(DST, "all_masks.npy"), mask[:, keep])
print("wrote %s  %.2f GB" % (DST, out.nbytes / 1e9))
