#!/usr/bin/env python3
"""Full-instrument architecture-diversity read: CoAtNet 5-fold OOF vs ConvNeXt 5-fold OOF vs their rank-mean blend. Pure numpy."""
import numpy as np
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CO = {0: "LR96_CTRL", 1: "LR96_F1", 2: "LR96_F2", 3: "LR96_F3", 4: "LR96_F4"}
CN = {0: "CNXT_F0", 1: "CNXT_F1", 2: "CNXT_F2", 3: "CNXT_F3", 4: "CNXT_F4"}


def load(tag):
    d = np.load("oof/oof_%s.npz" % tag, allow_pickle=True)
    return np.asarray(d["ids"], dtype=object).astype(str), d["pred"].astype(float), d["truth"].astype(float)


def macro(P, Y, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    a = [roc_auc_score((Y[idx, j] >= 0.5).astype(int), P[idx, j])
         for j in range(Y.shape[1]) if len(set((Y[idx, j] >= 0.5).astype(int).tolist())) > 1]
    return float(np.mean(a)) if a else float("nan")


def ranks(c):
    return np.argsort(np.argsort(c)).astype(float)


# assemble each arch's OOF across folds, aligned by id
ids_all, Pco_all, Pcn_all, Y_all = [], [], [], []
for f in range(5):
    ic, pc, yc = load(CO[f])
    icn, pcn, _ = load(CN[f])
    order = [ {s: i for i, s in enumerate(icn)}[s] for s in ic ]
    ids_all.append(ic); Pco_all.append(pc); Pcn_all.append(pcn[order]); Y_all.append(yc)
ids = np.concatenate(ids_all)
assert len(ids) == len(set(ids.tolist())), "duplicate ids across folds"
Pco = np.concatenate(Pco_all); Pcn = np.concatenate(Pcn_all); Y = np.concatenate(Y_all)

# rank-mean per finding, computed within the full pooled set
Pbl = np.zeros_like(Pco)
for j in range(12):
    Pbl[:, j] = (ranks(Pco[:, j]) + ranks(Pcn[:, j])) / 2.0

mco, mcn, mbl = macro(Pco, Y), macro(Pcn, Y), macro(Pbl, Y)
strong = max(mco, mcn)
print("OOF studies:", len(Y))
print("  CoAtNet 5-fold macro : %.4f" % mco)
print("  ConvNeXt 5-fold macro: %.4f" % mcn)
print("  rank-mean blend      : %.4f   (%+.4f vs stronger single-arch)" % (mbl, mbl - strong))

n = len(Y); rng = np.random.default_rng(0)
d = np.empty(2000)
for i in range(2000):
    idx = rng.integers(0, n, n)
    d[i] = macro(Pbl, Y, idx) - macro(Pco, Y, idx)
print("  blend vs CoAtNet     : mean %+.4f, win-rate %.1f%% (2000 paired bootstraps)"
      % (d.mean(), 100 * (d > 0).mean()))
print("\nper-finding (CoAtNet / ConvNeXt / blend):")
for j, name in enumerate(LAB):
    y = (Y[:, j] >= 0.5).astype(int)
    if len(set(y.tolist())) < 2:
        continue
    print("  %-18s %.4f / %.4f / %.4f" % (name, roc_auc_score(y, Pco[:, j]),
                                          roc_auc_score(y, Pcn[:, j]), roc_auc_score(y, Pbl[:, j])))
