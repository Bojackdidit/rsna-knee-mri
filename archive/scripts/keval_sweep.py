#!/usr/bin/env python3
"""Sweep K_EVAL on a trained arm. Zero submits, zero training.

The question the board cannot answer cheaply: how much accuracy does an arm lose when it is given
FEWER windows at inference? Model cost is linear in window count, so this curve is the whole runtime
story - 1322 studies x 48 windows = 63,456 forward passes per arm, and cutting to 12 cuts the
dominant cost 4x.

Note our arms are TRAINED with k=6 windows per study and then inferred at 48, so lower K_EVAL moves
inference TOWARD the training distribution.

Two kinds of arm are worth sweeping and they answer different things:
  - a GATE arm (trained with --group_fold 0 held out): the 631 studies are unseen, so both the
    absolute number and the shape are honest;
  - a SHIPPED arm (trained on everything but gold): the 631 studies were in its training set, so the
    ABSOLUTE number is contaminated and meaningless - but the SHAPE across K is still informative,
    because contamination applies equally at every K.
Usage: keval_sweep.py ARM.pt [K1,K2,...]
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import finetune_box as F

ARM = sys.argv[1]
KS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "48,24,12,9,6").split(",")]
LABELS = os.environ.get("SWEEP_LABELS", "/workspace/knee/labels_cd_ext.parquet")
GROUP = os.environ.get("SWEEP_GROUP", "/workspace/knee/study_fingerprint.csv")
DATA = F.DATA
LAB = F.LAB

ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {u: i for i, u in enumerate(ids)}
idset = set(ids)
tr = pd.read_csv(os.path.join(F.RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = [u for u in gold_df.index if u in idset]
soft = pd.read_parquet(LABELS); soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
goldset = set(gold_ids)
train_ids = [u for u in ids if u in soft.index and u not in goldset]

gdf = pd.read_csv(GROUP); gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
sizes = {}
for u in train_ids:
    f = fp_of.get(u)
    if f is not None:
        sizes[f] = sizes.get(f, 0) + 1
load = [0] * 5; fold_of_fp = {}
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    j = int(np.argmin(load)); fold_of_fp[f] = j; load[j] += n
# Fold is parameterised, not hardcoded. Scoring a fold-k arm against fold 0 grades it on its
# own training data - that is what made E0 read 0.9536 against a true 0.8961.
_FOLD = int(__import__('os').environ.get('GE_FOLD', '0'))
held = {f for f, j in fold_of_fp.items() if j == _FOLD}
val_ids = [u for u in train_ids if fp_of.get(u) in held]
labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}
print("[sweep] %d validation studies (grouped fold 0)" % len(val_ids), flush=True)

ck = torch.load(ARM, map_location="cpu", weights_only=False)
arch, res = ck.get("arch"), int(ck.get("res", 384))
bb = F.build_backbone(arch, pretrained=False)
model = F.RaptorClassifier(bb, F_dim=bb.num_features)
model.load_state_dict(ck["model"], strict=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)
print("[sweep] %s | arch %s | res %d" % (os.path.basename(ARM), arch, res), flush=True)

out = {}
for k in KS:
    ds = F.StudyWindows(DATA, val_ids, id2row, labels, res, k, train=False, norm="imagenet")
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=6, collate_fn=F.collate)
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    a = []
    for j in range(Y.shape[1]):
        yb = (Y[:, j] >= 0.5).astype(int)
        if len(set(yb.tolist())) > 1:
            a.append(roc_auc_score(yb, P[:, j]))
    out[k] = float(np.mean(a))
    np.savez(os.path.join(os.path.dirname(ARM) or ".", "sweep_%s_k%d.npz" % (os.path.basename(ARM).replace(".pt",""), k)),
             pred=P, truth=Y, ids=np.array(val_ids))
    print("  K_EVAL %3d -> macro-AUC %.4f" % (k, out[k]), flush=True)
base = out.get(max(KS))
print("\nK   macroAUC   delta vs K=%d" % max(KS))
for k in sorted(out, reverse=True):
    print("%3d  %.4f   %+.4f" % (k, out[k], out[k] - base))
