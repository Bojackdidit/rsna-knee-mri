#!/usr/bin/env python3
"""Fold-0 architecture-diversity read on the OOF instrument.

Compares the CoAtNet fold-0 arm (oof_LR96_CTRL) and the ConvNeXt fold-0 arm (oof_CNXT_F0) on the
IDENTICAL 870 held-out fold-0 studies. Both predictions are genuinely out-of-fold, so a rank-mean of
them is an honest estimate of the architecture-diverse ensemble on unseen data. Reports:
  - each arm's macro-AUC and the rank-mean blend's macro-AUC on the 870,
  - mean per-finding Spearman rho between the two arms (lower = more decorrelated = more to gain),
  - paired bootstrap: how often the blend beats the stronger single arm.
Pure numpy (no scipy dependency).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def macro(P, Y, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    a = [roc_auc_score((Y[idx, j] >= 0.5).astype(int), P[idx, j])
         for j in range(Y.shape[1]) if len(set((Y[idx, j] >= 0.5).astype(int).tolist())) > 1]
    return float(np.mean(a)) if a else float("nan")


def ranks(col):
    return np.argsort(np.argsort(col)).astype(float)


def rank_mean(A, B):
    R = np.zeros_like(A, dtype=float)
    for j in range(A.shape[1]):
        R[:, j] = (ranks(A[:, j]) + ranks(B[:, j])) / 2.0
    return R


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


ca = np.load("oof/oof_LR96_CTRL.npz", allow_pickle=True)
cn = np.load("oof/oof_CNXT_F0.npz", allow_pickle=True)
ca_ids = np.asarray(ca["ids"], dtype=object).astype(str)
cn_ids = np.asarray(cn["ids"], dtype=object).astype(str)
assert set(ca_ids) == set(cn_ids), "fold-0 study sets differ between the two arms"
posc = {s: i for i, s in enumerate(cn_ids)}
order = [posc[s] for s in ca_ids]
Pca, Y = ca["pred"].astype(float), ca["truth"].astype(float)
Pcn = cn["pred"].astype(float)[order]
Pbl = rank_mean(Pca, Pcn)

mca, mcn, mbl = macro(Pca, Y), macro(Pcn, Y), macro(Pbl, Y)
strong = max(mca, mcn)
print("fold-0 held-out studies:", len(Y))
print("  CoAtNet f0 macro   : %.4f" % mca)
print("  ConvNeXt f0 macro  : %.4f" % mcn)
print("  rank-mean blend    : %.4f   (%+.4f vs stronger single)" % (mbl, mbl - strong))
rhos = [spearman(Pca[:, j], Pcn[:, j]) for j in range(12)]
print("  mean per-finding rho: %.3f   (CoAtNet-sibling rho ~0.96; cross-arch ~0.88-0.90 is the win)"
      % float(np.mean(rhos)))

n = len(Y)
rng = np.random.default_rng(0)
d = np.empty(2000)
for i in range(2000):
    idx = rng.integers(0, n, n)
    d[i] = macro(Pbl, Y, idx) - macro(Pca, Y, idx)
print("  blend vs CoAtNet   : mean %+.4f, win-rate %.1f%% (2000 paired bootstraps on 870)"
      % (d.mean(), 100 * (d > 0).mean()))
print("\nper-finding (CoAtNet / ConvNeXt / blend / rho):")
for j, name in enumerate(LAB):
    y = (Y[:, j] >= 0.5).astype(int)
    if len(set(y.tolist())) < 2:
        continue
    print("  %-18s %.4f / %.4f / %.4f   rho %.2f"
          % (name, roc_auc_score(y, Pca[:, j]), roc_auc_score(y, Pcn[:, j]),
             roc_auc_score(y, Pbl[:, j]), rhos[j]))
