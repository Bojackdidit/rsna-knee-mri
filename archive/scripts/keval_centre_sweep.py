#!/usr/bin/env python3
"""At LOW window counts, does WHERE the windows sit recover the accuracy that cutting them costs?

K_EVAL sweep on the champion's own arm showed the shape: K=24 costs 0.0008, K=12 costs 0.0078,
K=9 costs 0.0113. Runtime is linear in K, and the bar is <=8 min total, i.e. model <=4.3 min, i.e.
K<=10. So the 0.008-0.011 loss at K=9-12 is exactly what has to be recovered.

Eval windows are currently spread by linspace across the WHOLE valid slice range. With 48 of them
that covers everything; with 9 it spends most of them on the ends of the stack. Concentrating the few
windows we can afford on the middle costs nothing at inference and needs no retraining.
"""
import os, sys, itertools
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import finetune_box as F

ARM = sys.argv[1]
KS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "12,9").split(",")]
CFS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1.0,0.75,0.5,0.35").split(",")]
LABELS = os.environ.get("SWEEP_LABELS", "/workspace/knee/labels_cd_ext.parquet")
GROUP = os.environ.get("SWEEP_GROUP", "/workspace/knee/study_fingerprint.csv")
DATA, LAB = F.DATA, F.LAB

ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {u: i for i, u in enumerate(ids)}
idset = set(ids)
tr = pd.read_csv(os.path.join(F.RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = [u for u in gold_df.index if u in idset]
soft = pd.read_parquet(LABELS); soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
train_ids = [u for u in ids if u in soft.index and u not in set(gold_ids)]
gdf = pd.read_csv(GROUP); gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
sizes = {}
for u in train_ids:
    f = fp_of.get(u)
    if f is not None: sizes[f] = sizes.get(f, 0) + 1
load = [0] * 5; fold_of_fp = {}
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    j = int(np.argmin(load)); fold_of_fp[f] = j; load[j] += n
# Fold is parameterised, not hardcoded. Scoring a fold-k arm against fold 0 grades it on its
# own training data - that is what made E0 read 0.9536 against a true 0.8961.
_FOLD = int(__import__('os').environ.get('GE_FOLD', '0'))
held = {f for f, j in fold_of_fp.items() if j == _FOLD}
val_ids = [u for u in train_ids if fp_of.get(u) in held]
labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}

ck = torch.load(ARM, map_location="cpu", weights_only=False)
arch, res = ck.get("arch"), int(ck.get("res", 384))
bb = F.build_backbone(arch, pretrained=False)
model = F.RaptorClassifier(bb, F_dim=bb.num_features)
model.load_state_dict(ck["model"], strict=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)
print("[sweep] %s | %s | res %d | %d val studies" % (os.path.basename(ARM), arch, res, len(val_ids)), flush=True)

def run(k, cf):
    F.CENTRE_FRAC = cf
    ds = F.StudyWindows(DATA, val_ids, id2row, labels, res, k, train=False, norm="imagenet")
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=6, collate_fn=F.collate)
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    a = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j])
         for j in range(Y.shape[1]) if len(set(((Y[:, j] >= 0.5).astype(int)).tolist())) > 1]
    return float(np.mean(a))

print("%5s %8s %10s" % ("K", "centre", "macroAUC"), flush=True)
best = {}
for k, cf in itertools.product(KS, CFS):
    v = run(k, cf)
    best[(k, cf)] = v
    print("%5d %8.2f %10.4f" % (k, cf, v), flush=True)
print("\nbest per K:")
for k in KS:
    r = sorted(((v, cf) for (kk, cf), v in best.items() if kk == k), reverse=True)
    print("  K=%2d -> %.4f at centre_frac %.2f" % (k, r[0][0], r[0][1]))
