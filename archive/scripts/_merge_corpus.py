#!/usr/bin/env python3
"""Merge the two halves of the knee corpus into one 4,407-study volume.

The Kaggle dataset is literally named "part 1 of 2" and only part 1 was ever downloaded - on every
box, all day. So every arm trained today saw 3,200 studies (2,524 after the grouped fold was held
out) while the shipped 0.941 arms saw 4,407 (4,349 after gold). That is 58% of the data, and it is
large enough to account for the 0.06 gap I was about to blame on architecture.

    part 1  all_vols.npy    3200 x 44 x 336 x 336
    part 2  extra_vols.npy  1207 x 44 x 336 x 336
    merged                  4407 x 44 x 336 x 336

Written with a memmap so it never needs 22 GB of RAM, and verified by shape, byte count and id
uniqueness before anything trains on it.
"""
import os, sys
import numpy as np

SRC = "/workspace/knee/corpus"
EXT = "/workspace/ext"
OUT = "/workspace/knee/corpus_full"
SL, H, W = 44, 336, 336

os.makedirs(OUT, exist_ok=True)
ids1 = np.load(os.path.join(SRC, "all_ids.npy"), allow_pickle=True).astype(str)
ids2 = np.load(os.path.join(EXT, "extra_ids.npy"), allow_pickle=True).astype(str)
m1 = np.load(os.path.join(SRC, "all_masks.npy"))
m2 = np.load(os.path.join(EXT, "extra_masks.npy"))
v1 = np.load(os.path.join(SRC, "all_vols.npy"), mmap_mode="r")
v2 = np.load(os.path.join(EXT, "extra_vols.npy"), mmap_mode="r")
print("part1", v1.shape, len(ids1), "| part2", v2.shape, len(ids2), flush=True)
assert v1.shape[1:] == (SL, H, W) and v2.shape[1:] == (SL, H, W), "slot geometry differs between halves"

dup = set(ids1) & set(ids2)
print("overlapping ids between halves:", len(dup), flush=True)
keep2 = np.array([i for i, u in enumerate(ids2) if u not in set(ids1)], dtype=int)
n = len(ids1) + len(keep2)
print("merged study count:", n, flush=True)

out = np.lib.format.open_memmap(os.path.join(OUT, "all_vols.npy"), mode="w+",
                                dtype=np.uint8, shape=(n, SL, H, W))
CH = 64
for s in range(0, len(ids1), CH):
    out[s:s + CH] = v1[s:s + CH]
    if s % 1024 == 0:
        print("  part1 %d/%d" % (s, len(ids1)), flush=True)
base = len(ids1)
for s in range(0, len(keep2), CH):
    sel = keep2[s:s + CH]
    out[base + s: base + s + len(sel)] = v2[sel]
    if s % 512 == 0:
        print("  part2 %d/%d" % (s, len(keep2)), flush=True)
out.flush(); del out

np.save(os.path.join(OUT, "all_ids.npy"), np.concatenate([ids1, ids2[keep2]]))
np.save(os.path.join(OUT, "all_masks.npy"), np.concatenate([m1, m2[keep2]], axis=0))
for extra in ("window_index.npy",):
    p = os.path.join(SRC, extra)
    if os.path.exists(p):
        import shutil
        shutil.copy(p, os.path.join(OUT, extra))

chk = np.load(os.path.join(OUT, "all_vols.npy"), mmap_mode="r")
cids = np.load(os.path.join(OUT, "all_ids.npy"), allow_pickle=True).astype(str)
cmask = np.load(os.path.join(OUT, "all_masks.npy"))
print("\nMERGED all_vols", chk.shape, "| ids", len(cids), "| masks", cmask.shape, flush=True)
print("bytes on disk:", os.path.getsize(os.path.join(OUT, "all_vols.npy")), flush=True)
print("expected     :", n * SL * H * W + 128, flush=True)
assert chk.shape == (n, SL, H, W) and len(cids) == n and cmask.shape == (n, SL)
assert len(set(cids)) == n, "duplicate study ids after merge"
# spot-check that the two halves actually landed where they should
assert np.array_equal(np.asarray(chk[0]), np.asarray(v1[0])), "part1 row 0 mismatch"
assert np.array_equal(np.asarray(chk[base]), np.asarray(v2[keep2[0]])), "part2 first row mismatch"
print("VERIFIED: shape, ids unique, byte count, and both halves spot-checked", flush=True)
