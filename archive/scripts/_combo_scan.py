#!/usr/bin/env python3
"""Every pair and trio of the arms we already hold, rank-blended, on the real-gold ruler. No GPU.

Rank-mean is the operation the kernel actually performs, so the offline number and the shipped
operation are the same thing. Restricted to dumps covering all 58 gold studies so every combination
is compared on identical rows.

An arm is only SHIPPABLE if its checkpoint still exists - several of the best historical adders were
lost with destroyed boxes (memory/pull-weights-before-teardown), and a blend built on a missing .pt
is a result that cannot be submitted.
"""
import glob, itertools, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
tr = pd.read_csv("train.csv", encoding="latin-1")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = tr[tr[LAB].notna().all(axis=1)]
GY = {u: (gold.loc[gold["StudyInstanceUID"] == u, LAB].values[0] >= 1).astype(int)
      for u in gold["StudyInstanceUID"]}
IDS = sorted(GY)
Y = np.stack([GY[u] for u in IDS])

# find a checkpoint for each tag, anywhere in the tree
CK = {}
for p in glob.glob("**/raptor_ft_*.pt", recursive=True):
    CK.setdefault(os.path.basename(p)[10:-3], p)

arms = {}
for p in sorted(glob.glob("**/raptor_gold_*.npz", recursive=True)):
    try:
        z = np.load(p, allow_pickle=True)
        ids = np.array([str(x) for x in z["ids"]])
        pred = np.asarray(z["pred"], dtype=float)
    except Exception:
        continue
    if pred.ndim != 2 or pred.shape[1] != 12 or len(ids) != len(pred):
        continue
    pos = {u: i for i, u in enumerate(ids)}
    if not set(IDS) <= set(pos):
        continue
    tag = os.path.basename(p)[12:-4]
    if tag in arms:
        continue
    arms[tag] = pred[[pos[u] for u in IDS]]
print("arms covering all %d gold studies: %d" % (len(IDS), len(arms)))
print("of those, with a checkpoint on disk: %d" % sum(1 for t in arms if t in CK))


def macro(P):
    a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(12) if len(set(Y[:, j].tolist())) > 1]
    return float(np.mean(a))


def rk(P):
    return np.stack([rankdata(P[:, j]) / len(P) for j in range(12)], 1)


RK = {t: rk(P) for t, P in arms.items()}
single = {t: macro(P) for t, P in arms.items()}
SHIP = ("confdist_swa", "coatnet_w80_full_swa")
base = macro(np.mean([RK[SHIP[0]], RK[SHIP[1]]], 0))
print("SHIPPED PAIR %s + %s -> %.4f on this ruler (board 0.941)" % (SHIP[0], SHIP[1], base))
print()

res = []
for a, b in itertools.combinations(sorted(arms), 2):
    res.append((macro(np.mean([RK[a], RK[b]], 0)), (a, b)))
res.sort(reverse=True)
print("TOP PAIRS (* = both checkpoints present, so actually shippable)")
shown = 0
for s, (a, b) in res:
    ok = a in CK and b in CK
    print("  %.4f %s %-26s + %-26s  (solo %.4f / %.4f)"
          % (s, "*" if ok else " ", a, b, single[a], single[b]))
    shown += 1
    if shown >= 14:
        break
print()
best_ship = next(((s, c) for s, c in res if c[0] in CK and c[1] in CK), None)
if best_ship:
    print("BEST SHIPPABLE PAIR %.4f = %s + %s   (%+.4f vs the shipped pair)"
          % (best_ship[0], best_ship[1][0], best_ship[1][1], best_ship[0] - base))
    for t in best_ship[1]:
        print("    %-26s %s" % (t, CK[t]))
pd.DataFrame([{"macro": s, "a": c[0], "b": c[1], "shippable": c[0] in CK and c[1] in CK}
              for s, c in res]).to_csv("_combo_pairs.csv", index=False)
print("wrote _combo_pairs.csv (%d pairs)" % len(res))

# --- trios and quads, SHIPPABLE arms only -------------------------------------------------
have = sorted([t for t in arms if t in CK])
print()
print("shippable arms (%d): %s" % (len(have), ", ".join(have)))
for k in (3, 4):
    r = []
    for c in itertools.combinations(have, k):
        r.append((macro(np.mean([RK[t] for t in c], 0)), c))
    r.sort(reverse=True)
    print()
    print("TOP %d-ARM COMBOS (shipped pair = %.4f)" % (k, base))
    for s, c in r[:6]:
        print("  %.4f  %+.4f  %s" % (s, s - base, " + ".join(c)))
