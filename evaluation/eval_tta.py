#!/usr/bin/env python3
"""Screen test-time augmentation on the SCANNER-DISJOINT fold. No board submit needed.

TTA has never been submitted on this competition - checked all 50 entries, none mention it - and
unlike every other AUC idea on the closed list it needs NO retraining. This screens it honestly:
the gate arm was trained with grouped fold 0 excluded, so fold 0 is a clean held-out set for it.

The TTA ops are chosen to preserve LATERALITY, which is signal here (5 of 12 findings are
side-specific and the model resolves side correctly - see the laterality refutation). So NO
horizontal flip. Slice-order reversal flips the DEPTH axis (the 3 consecutive slices that form the
RGB channels), not left-right, and scale/gamma are geometry-free.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.argv = [sys.argv[0]]                      # keep finetune_box's argparse out of the way
import finetune_box as FB

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="raptor_ft_gate_base.pt")
ap.add_argument("--group_csv", default="study_fingerprint.csv")
ap.add_argument("--group_fold", type=int, default=0)
ap.add_argument("--labels", default="labels_cd_ext.parquet")
ap.add_argument("--res", type=int, default=224); ap.add_argument("--k_eval", type=int, default=12)
ap.add_argument("--bs", type=int, default=4);   ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])
a = ap.parse_args([])

LAB = FB.LAB
ids = np.load(os.path.join(FB.DATA, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {u: i for i, u in enumerate(ids)}
tr = pd.read_csv(os.path.join(FB.RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = set(tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"])
soft = pd.read_parquet(a.labels); soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
train_ids = [u for u in ids if u in soft.index and u not in gold]

g = pd.read_csv(a.group_csv); fp = dict(zip(g.StudyInstanceUID.astype(str), g.fp.astype(str)))
sizes = {}
for u in train_ids:
    f = fp.get(u)
    if f is not None: sizes[f] = sizes.get(f, 0) + 1
load = [0] * 5; fold_of = {}
for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    j = int(np.argmin(load)); fold_of[f] = j; load[j] += n
held = {f for f, j in fold_of.items() if j == a.group_fold}
val_ids = [u for u in train_ids if fp.get(u) in held]
labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}
print("scanner-disjoint val: %d studies, %d fingerprints" % (len(val_ids), len(held)), flush=True)

ck = torch.load(a.ckpt, map_location="cpu")
dev = "cuda" if torch.cuda.is_available() else "cpu"
sd = ck["model"] if "model" in ck else ck
arch = ck.get("arch", "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k")
_bb = FB.build_backbone(arch, pretrained=False)
# F_dim is the BACKBONE's feature width (1024 for coatnet_rmlp_2_rw_224), not the label count.
_fdim = int(getattr(_bb, "num_features", 0)) or int(sd["clsW"].shape[1])
model = FB.RaptorClassifier(_bb, F_dim=_fdim, n=len(LAB)).to(dev)
model.load_state_dict(sd); model.eval()
print("loaded %s (arch %s)" % (a.ckpt, arch), flush=True)

ds = FB.StudyWindows(FB.DATA, val_ids, id2row, labels, a.res, a.k_eval, train=False, norm="imagenet")
dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=FB.collate)

OPS = {
    "identity":      lambda x: x,
    "slice_reverse": lambda x: x.flip(2),                                    # depth flip, NOT left-right
    "scale_0.92":    lambda x: F.interpolate(F.interpolate(x, scale_factor=0.92, mode="bilinear",
                                   align_corners=False, recompute_scale_factor=False),
                                   size=x.shape[-2:], mode="bilinear", align_corners=False),
    "gamma_0.9":     lambda x: x.clamp(1e-4, None).pow(0.9),
}

@torch.no_grad()
def run():
    P = {k: [] for k in OPS}; Y = []
    for x, y in dl:
        B, K = x.shape[0], x.shape[1]
        xf = x.view(B * K, *x.shape[2:]).to(dev)
        for name, op in OPS.items():
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(op(xf).view(B, K, *x.shape[2:])).float())
            P[name].append(o.cpu().numpy())
        Y.append(y.numpy())
    return {k: np.concatenate(v) for k, v in P.items()}, np.concatenate(Y)

P, Y = run()
def macro(p):
    aucs = []
    for j in range(len(LAB)):
        yb = (Y[:, j] >= 0.5).astype(int)
        if len(set(yb.tolist())) > 1: aucs.append(roc_auc_score(yb, p[:, j]))
    return float(np.mean(aucs))

base = macro(P["identity"])
print("\n  %-26s %s" % ("scheme", "macro-AUC   delta"))
print("  %-26s %.4f" % ("identity (no TTA)", base))
for k in OPS:
    if k == "identity": continue
    print("  %-26s %.4f   %+.4f" % (k + " alone", macro(P[k]), macro(P[k]) - base))
for combo in [("identity", "slice_reverse"), ("identity", "scale_0.92"),
              ("identity", "slice_reverse", "scale_0.92"),
              ("identity", "slice_reverse", "scale_0.92", "gamma_0.9")]:
    avg = np.mean([P[c] for c in combo], axis=0)
    print("  %-26s %.4f   %+.4f" % ("avg(" + "+".join(c[:6] for c in combo) + ")", macro(avg), macro(avg) - base))
