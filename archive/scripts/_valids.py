#!/usr/bin/env python3
"""Reproduce the exact validation id list for a given (--group_fold, --reclaim) without training.

E0 launched before the trainer recorded its own val ids, and it uses --reclaim 0.7 while E1/E1b do
not - so E0 holds out 350 studies and E1 holds out all 870 of the same fold. Their reported bests are
NOT comparable: different denominators, and E0 trained on 520 of the studies E1 validates on.

The split is fully deterministic (greedy largest-fingerprint-first fold assignment, then a fixed
seed 20260912 shuffle for the reclaim draw), so it can be regenerated exactly. This replicates that
code path verbatim. Usage: _valids.py FOLD RECLAIM OUT.csv
"""
import os, sys
import numpy as np, pandas as pd
import finetune_box as F

FOLD = int(sys.argv[1]); RECLAIM = float(sys.argv[2]); OUT = sys.argv[3]
FOLDS = int(os.environ.get("VAL_FOLDS", "5"))
LABELS = os.environ.get("VAL_LABELS", "/workspace/knee/labels_opus_blend.parquet")
GCSV = os.environ.get("VAL_GROUP_CSV", "/workspace/knee/study_fingerprint.csv")

ids = np.load(os.path.join(F.DATA, "all_ids.npy"), allow_pickle=True).astype(str)
tr = pd.read_csv(os.path.join(F.RSNA, "train.csv"))
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[F.LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
goldset = set(u for u in gold_df.index if u in set(ids))

soft = pd.read_parquet(LABELS)
soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
train_ids = [u for u in ids if u in soft.index and u not in goldset]

gdf = pd.read_csv(GCSV); gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
sizes = {}
for u in train_ids:
    f = fp_of.get(u)
    if f is not None:
        sizes[f] = sizes.get(f, 0) + 1
load = [0] * FOLDS
fold_of_fp = {}
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    j = int(np.argmin(load)); fold_of_fp[f] = j; load[j] += n
held = {f for f, j in fold_of_fp.items() if j == FOLD}
val = [u for u in train_ids if fp_of.get(u) in held]
tids = [u for u in train_ids if fp_of.get(u) not in held]
print("folds %s | fold %d holds %d fingerprints, %d studies" % (load, FOLD, len(held), len(val)))

if RECLAIM > 0.0:
    _hf = sorted(held)
    _rng = np.random.default_rng(20260912); _rng.shuffle(_hf)
    _size = {}
    for _u in val:
        _f = fp_of.get(_u); _size[_f] = _size.get(_f, 0) + 1
    _tot = sum(_size.values()); _floor = (1.0 - RECLAIM) * _tot
    _back, _acc = set(), 0
    for _f in _hf:
        if (_tot - _acc - _size.get(_f, 0)) < _floor:
            continue
        _back.add(_f); _acc += _size.get(_f, 0)
    _keep = set(_hf) - _back
    moved = [u for u in val if fp_of.get(u) in _back]
    val = [u for u in val if fp_of.get(u) in _keep]
    tids = tids + moved
    print("reclaim %.2f -> returned %d studies, validate on %d" % (RECLAIM, len(moved), len(val)))
    assert not ({fp_of.get(u) for u in tids} & {fp_of.get(u) for u in val}), "scanner leaked"

pd.DataFrame({"StudyInstanceUID": val}).to_csv(OUT, index=False)
print("wrote %s (%d studies)" % (OUT, len(val)))
