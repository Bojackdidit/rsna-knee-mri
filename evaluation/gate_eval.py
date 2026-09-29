#!/usr/bin/env python3
"""Score ANY arm on the grouped fold and dump its predictions. Usage: gate_eval.py ARM.pt TAG"""
import os, sys
import json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import finetune_box as F

ARM, TAG = sys.argv[1], sys.argv[2]
K = int(os.environ.get("GE_K", "12"))
LABELS = os.environ.get("GE_LABELS", "/workspace/knee/labels_opus_blend.parquet")
GROUP = os.environ.get("GE_GROUP", "/workspace/knee/study_fingerprint.csv")
DATA, LAB = F.DATA, F.LAB

ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {u: i for i, u in enumerate(ids)}
tr = pd.read_csv(os.path.join(F.RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_ids = set(tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"]) & set(ids)
soft = pd.read_parquet(LABELS); soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
train_ids = [u for u in ids if u in soft.index and u not in gold_ids]
gdf = pd.read_csv(GROUP); gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
sizes = {}
for u in train_ids:
    f = fp_of.get(u)
    if f is not None: sizes[f] = sizes.get(f, 0) + 1
load = [0] * 5; fold_of = {}
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    j = int(np.argmin(load)); fold_of[f] = j; load[j] += n
_IDS = os.environ.get("GE_IDS", "")
if _IDS and os.path.exists(_IDS):
    _keep = set(x.strip() for x in open(_IDS).read().split() if x.strip() and x.strip() != "StudyInstanceUID")
    held = None
    _FOLD = -1   # explicit id list, not a grouped fold; -1 marks that in the dump metadata
else:
    _keep = None
    _FOLD = int(os.environ.get("GE_FOLD", "0"))
    held = {f for f, j in fold_of.items() if j == _FOLD}
if _keep is not None and not (set(_keep) & set(train_ids)):
    _real = tr.set_index("StudyInstanceUID")
    val_ids = [u for u in ids if u in _keep and u in _real.index
               and _real.loc[u, LAB].notna().all()]
    labels = {u: _real.loc[u, LAB].values.astype(np.float32) for u in val_ids}
    print("[gate] REAL image-derived labels from train.csv on %d studies" % len(val_ids), flush=True)
elif _keep is not None:
    val_ids = [u for u in train_ids if u in _keep]
else:
    val_ids = [u for u in train_ids if fp_of.get(u) in held]
assert val_ids, "no validation studies selected - check GE_IDS/GE_FOLD"
labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}
print("[gate] %d held-out studies, fold sizes %s" % (len(val_ids), load), flush=True)

ck = torch.load(ARM, map_location="cpu", weights_only=False)
arch, res = ck.get("arch"), int(ck.get("res", 384))
bb = F.build_backbone(arch, pretrained=False)
model = F.RaptorClassifier(bb, F_dim=bb.num_features)
missing, unexpected = model.load_state_dict(ck["model"], strict=False)
_POOL = os.environ.get("GE_POOL", "")
if _POOL:
    assert _POOL in ("attn", "mean", "max"), "GE_POOL must be attn|mean|max"
    model.pool = _POOL
    print("[gate] POOLING OVERRIDDEN -> %s (was %s)" % (_POOL, ck.get("pool", "attn")), flush=True)
if missing or unexpected:
    print("[gate] state_dict missing=%s unexpected=%s" % (list(missing)[:4], list(unexpected)[:4]), flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)
print("[gate] %s | %s | res %d | K %d" % (os.path.basename(ARM), arch, res, K), flush=True)

# GE_SLOTS unset -> use the checkpoint; 'all' -> every slot; '0,2,4' -> exactly those.
_env = os.environ.get("GE_SLOTS", None)
if _env is None:
    _slots = ck.get("slots", None)
elif _env.strip().lower() in ("all", "none", ""):
    _slots = None
else:
    _slots = [int(x) for x in _env.split(",") if x.strip() != ""]
F.KEEP_SLOTS = list(_slots) if _slots else None
print("[gate] slot source: %s" % ("checkpoint" if _env is None else "GE_SLOTS=%r" % _env), flush=True)
F.CENTRE_FRAC = float(ck.get("centre_frac", os.environ.get("GE_CENTRE_FRAC", 1.0)))
F.CROP_FRAC = float(ck.get("crop_frac", os.environ.get("GE_CROP_FRAC", 1.0)))
F.SLICE_DILATE = int(ck.get("slice_dilate", os.environ.get("GE_SLICE_DILATE", 1)))
print("[gate] geometry: slots=%s centre_frac=%.2f crop_frac=%.2f dilate=%d"
      % (F.KEEP_SLOTS, F.CENTRE_FRAC, F.CROP_FRAC, F.SLICE_DILATE), flush=True)
_PAR = os.environ.get("GE_PARITY", "")
if _PAR != "":
    _par = int(_PAR)
    _orig = F.StudyWindows._centers
    def _half(self, valid, _o=_orig, _p=_par):
        _k = self.k
        self.k = _k * 2
        try:
            cs = _o(self, valid)
        finally:
            self.k = _k
        return list(cs)[_p::2][:_k]
    F.StudyWindows._centers = _half
    print("[gate] INTERLEAVE: planning %d positions, taking parity %d -> %d windows"
          % (K * 2, _par, K), flush=True)
ds = F.StudyWindows(DATA, val_ids, id2row, labels, res, K, train=False, norm="imagenet")
dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=6, collate_fn=F.collate)
P, Y = [], []
with torch.no_grad():
    for b in dl:
        x, y = b[0], b[1]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = torch.sigmoid(model(x.to(dev)).float())
        P.append(o.cpu().numpy()); Y.append(y.numpy())
P, Y = np.concatenate(P), np.concatenate(Y)
aucs = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j]) for j in range(Y.shape[1])
        if len(set(((Y[:, j] >= 0.5).astype(int)).tolist())) > 1]
out = os.path.join(os.environ.get("GATE_OUT_DIR", "/workspace/knee"), "raptor_gold_%s.npz" % TAG)
_cfg = {"fold": _FOLD, "arm": os.path.basename(ARM),
        "arm_mtime": int(os.path.getmtime(ARM)), "arm_bytes": os.path.getsize(ARM),
        "slots": "all" if not F.KEEP_SLOTS else ",".join(str(x) for x in F.KEEP_SLOTS),
        "K": K, "res": res, "arch": str(arch), "n_val": len(val_ids),
        "centre_frac": F.CENTRE_FRAC, "crop_frac": F.CROP_FRAC, "slice_dilate": F.SLICE_DILATE,
        "labels": os.path.basename(LABELS), "ids_file": os.path.basename(_IDS) if _IDS else "",
        "parity": _PAR if _PAR != "" else "none", "pool": _POOL or "as-trained"}
print("[gate] config %s" % _cfg, flush=True)
np.savez(out, pred=P, truth=Y, ids=np.array(val_ids, dtype=object),
         cfg=np.array(json.dumps(_cfg), dtype=object))
print("[gate] %s macro-AUC %.4f over %d findings -> %s" % (TAG, float(np.mean(aucs)), len(aucs), out), flush=True)
