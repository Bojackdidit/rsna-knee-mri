#!/usr/bin/env python3
"""Two arms, INTERLEAVED window positions: K-cost runtime for 2K-worth of coverage.

Arms run CONCURRENTLY, one per GPU, so wall-clock is ONE arm's cost regardless of how many arms there
are (up to 2). Today they both read the SAME window positions, which wastes that: the blend sees K
distinct positions for K cost.

Give arm A the even positions of a 2K grid and arm B the odd ones, and the blend sees 2K distinct
positions for K cost. The accuracy lost by cutting K is a COVERAGE loss, so this should recover most
of it for free - no retraining, no extra runtime.

Compares, on the same 631 studies:
    both arms @K, same positions      (what we ship today, scaled down)
    both arms @K, interleaved         (the proposal)
    both arms @2K, same positions     (the accuracy target)
"""
import os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import finetune_box as F

ARMS = sys.argv[1].split(",")
KS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "12,9").split(",")]
LABELS = os.environ.get("SWEEP_LABELS", "/workspace/knee/labels_cd_ext.parquet")
GROUP = os.environ.get("SWEEP_GROUP", "/workspace/knee/study_fingerprint.csv")
DATA, LAB = F.DATA, F.LAB

ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {u: i for i, u in enumerate(ids)}
tr = pd.read_csv(os.path.join(F.RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = [u for u in gold_df.index if u in set(ids)]
soft = pd.read_parquet(LABELS); soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
train_ids = [u for u in ids if u in soft.index and u not in set(gold_ids)]
gdf = pd.read_csv(GROUP); gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
sizes = {}
for u in train_ids:
    f = fp_of.get(u)
    if f is not None: sizes[f] = sizes.get(f, 0) + 1
load = [0]*5; fold = {}
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    j = int(np.argmin(load)); fold[f] = j; load[j] += n
# Fold is parameterised, not hardcoded. Scoring a fold-k arm against fold 0 grades it on its
# own training data - that is what made E0 read 0.9536 against a true 0.8961.
_FOLD = int(__import__('os').environ.get('GE_FOLD', '0'))
held = {f for f, j in fold.items() if j == _FOLD}
val_ids = [u for u in train_ids if fp_of.get(u) in held]
labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}
print("[test] %d validation studies" % len(val_ids), flush=True)

_orig = F.StudyWindows._centers
PHASE = {"k2": None, "ph": None}
def patched(self, valid):
    if PHASE["k2"] is None:
        return _orig(self, valid)
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo+1, hi) if c-1 >= lo and c+1 <= hi]
    if not cs:
        cs = [max(1, min((lo+hi)//2, self._D-2))]
    idx = np.linspace(0, len(cs)-1, PHASE["k2"]).round().astype(int)
    sel = idx[PHASE["ph"]::2]
    return [cs[i] for i in sel]
F.StudyWindows._centers = patched

def load_arm(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    bb = F.build_backbone(ck.get("arch"), pretrained=False)
    m = F.RaptorClassifier(bb, F_dim=bb.num_features)
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.eval().to(dev), int(ck.get("res", 384)), dev

def predict(model, res, dev, k, k2=None, ph=None):
    PHASE["k2"], PHASE["ph"] = k2, ph
    ds = F.StudyWindows(DATA, val_ids, id2row, labels, res, k, train=False, norm="imagenet")
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=6, collate_fn=F.collate)
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
    PHASE["k2"] = None
    return np.concatenate(P), np.concatenate(Y)

def macro(P, Y):
    a = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j])
         for j in range(Y.shape[1]) if len(set(((Y[:, j] >= 0.5).astype(int)).tolist())) > 1]
    return float(np.mean(a))
def rk(P): return np.column_stack([rankdata(P[:, j])/len(P) for j in range(P.shape[1])])

mA, resA, dev = load_arm(ARMS[0]); mB, resB, _ = load_arm(ARMS[1])
for K in KS:
    pa, Y = predict(mA, resA, dev, K)
    pb, _ = predict(mB, resB, dev, K)
    same = macro((rk(pa)+rk(pb))/2, Y)
    ia, _ = predict(mA, resA, dev, K, k2=2*K, ph=0)
    ib, _ = predict(mB, resB, dev, K, k2=2*K, ph=1)
    inter = macro((rk(ia)+rk(ib))/2, Y)
    da, _ = predict(mA, resA, dev, 2*K)
    db, _ = predict(mB, resB, dev, 2*K)
    dbl = macro((rk(da)+rk(db))/2, Y)
    print("K=%2d | same-positions %.4f | INTERLEAVED %.4f | both@2K=%d %.4f | recovered %.0f%%"
          % (K, same, inter, 2*K, dbl, 100*(inter-same)/max(dbl-same, 1e-9)), flush=True)
