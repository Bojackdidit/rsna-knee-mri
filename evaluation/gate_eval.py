#!/usr/bin/env python3
"""Score ANY arm on the grouped fold and dump its predictions. Usage: gate_eval.py ARM.pt TAG

Why this exists: the best pair we can build offline is CROSS-FAMILY (CoAtNet + EfficientNet, 0.9201)
and every same-family pair is worse - CoAtNet/effv2s outputs correlate 0.8875 against 0.9357 for
CoAtNet/CoAtNet. The shipped champion pair is CoAtNet + CoAtNet. So the open question is whether
swapping ONE champion arm for an EfficientNet arm lifts the pair that actually scores 0.941.

That cannot be answered today because the champion arms have no gate dump on this fold, and the one
number we do have for them (0.9377 from the centre sweep) was graded against cd_ext labels while every
arm trained this week was graded against blend - and this gate measures AGREEMENT WITH ITS LABEL SET,
so those two numbers are not comparable. This re-scores an arm against the SAME blend labels and the
SAME fold as everything else, and saves pred/truth/ids so it can be paired with any other arm.

Fold construction is copied from finetune_box exactly (greedy balanced assignment over fingerprints,
biggest first into the lightest fold) so the 870 held-out studies are identical.
"""
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
# GE_IDS points at the valids_<tag>.csv the training run itself wrote, i.e. EXACTLY the studies
# that arm held out. Without it this function always scored against fold 0, which is correct only
# for fold-0 arms: a fold-1 arm trained on all of fold 0, so grading it there grades it on its own
# training data. E0 read 0.9536 that way against a true 0.8961 - a +0.057 illusion. Doing that to
# every fold arm would have produced a ranking that looked excellent and was worthless.
_IDS = os.environ.get("GE_IDS", "")
if _IDS and os.path.exists(_IDS):
    _keep = set(x.strip() for x in open(_IDS).read().split() if x.strip() and x.strip() != "StudyInstanceUID")
    held = None
    _FOLD = -1   # explicit id list, not a grouped fold; -1 marks that in the dump metadata
else:
    _keep = None
    _FOLD = int(os.environ.get("GE_FOLD", "0"))
    held = {f for f, j in fold_of.items() if j == _FOLD}
# The 58 gold studies carry REAL image-derived labels and the trainer excludes them from EVERY arm,
# champion included - so they are the only gate on which our new arms and the shipped champion can
# be compared honestly. They are absent from train_ids by construction, so an explicit GE_IDS list
# naming them must be allowed to select from ALL studies and to take its labels from train.csv
# rather than the distilled parquet. Small (58 studies, 9-35 positives per finding) so it is a
# DIAGNOSTIC, not a selection gate - it has predicted the board wrongly seven times.
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
# GE_POOL overrides how per-window features become a study score, WITHOUT retraining. An attention
# decomposition over 50 studies found the learned head is effectively a fixed per-finding plane
# prior: a frozen profile with zero image dependence reproduces its output at rho 0.966, and plain
# mean-pooling ranks studies at rho 0.970 against it. max-pooling IS a different ranking (rho 0.864)
# and the flattest-attention findings are the focal ones it should help. Free to test.
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

# Restore the geometry the arm was TRAINED on. StudyWindows reads module-level globals
# (KEEP_SLOTS, CENTRE_FRAC, CROP_FRAC, SLICE_DILATE) and importing finetune_box fresh leaves them
# at their defaults - so a 3-series arm was being evaluated on the full 5-series volume. Every
# finding came out systematically low (mean -0.0053 on E1b) against the number its own training
# run reported, on the same checkpoint and the same 870 studies.
# Prefer what the checkpoint records; fall back to GE_SLOTS for arms trained before it recorded it.
# PRECEDENCE: an explicitly set GE_SLOTS must OVERRIDE what the checkpoint records, or the
# geometry can never be tested. Recording geometry in the checkpoint (added earlier tonight) had
# the side effect of defeating the override: a deliberate 5-series probe silently re-ran the
# checkpoint's 3 series and returned the SAME number twice, which reads as 'no difference' rather
# than 'the test did not run'. GE_SLOTS unset -> use the checkpoint. GE_SLOTS='all' -> every slot.
# GE_SLOTS='0,2,4' -> exactly those.
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
# GE_PARITY reproduces what the KERNEL actually does. The kernel plans 2K window positions and
# hands arm 0 the even ones and arm 1 the odd (`xw[a % 2::2]`), so the pair covers 2K positions for
# K of work per arm. Nothing offline reproduced that, so every "pair = solo + 0.009" statement has
# been an assumption. With GE_K=24 GE_PARITY=0 for one arm and GE_PARITY=1 for the other, blending
# the two dumps gives the interleaved pair score as a MEASUREMENT.
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
# Stamp the ACTUAL configuration into the dump. Every silent failure tonight had the same shape:
# a run did something other than what was asked and its output looked like a valid result. A number
# is not interpretable without the configuration that produced it, so they travel together and any
# comparison can assert the two runs actually differed.
# fold is THE field whose absence made E0 read 0.9536 against a true 0.8961 - an arm graded on
# its own training data. A dump without it cannot be compared to another dump. Stamp it, plus
# enough arm identity to tell two checkpoints apart after the fact.
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
