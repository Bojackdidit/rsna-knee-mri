#!/usr/bin/env python3
"""Score 224-only combinations against the shipped 384 pair.

Only same-resolution arms can ship together: the kernel builds each study's windows ONCE and
asserts "single pass needs one resolution", so a 384 arm and a 224 arm cannot ride in the same
notebook. The 224 arms are therefore their own competing team, and the question is whether that
team as a whole beats the 384 pair that holds the board.

Bar = the shipped pair as this harness computes it (0.9331). Bootstrap win-rate resamples studies.
Self-contained on purpose - pairtest.py runs its report at import time.
"""
import glob, itertools, os
import numpy as np

SHIPPED_DIR, NEW_DIR = "ortho/artifacts", "goldnew"
SHIPPED = ("confdist_swa", "coatnet_w80_full_swa")
B = 2000


def auc(y, s):
    y = np.asarray(y).astype(int); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    o = s.argsort(kind="mergesort"); r = np.empty(len(s)); ss = s[o]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def macro(Y, P):
    v = [auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def rankify(P):
    R = np.empty_like(P, dtype=float)
    for j in range(P.shape[1]):
        s = P[:, j]; o = s.argsort(kind="mergesort"); r = np.empty(len(s)); ss = s[o]; i = 0
        while i < len(ss):
            k = i
            while k + 1 < len(ss) and ss[k + 1] == ss[i]:
                k += 1
            r[o[i:k + 1]] = (i + k) / 2.0 + 1.0; i = k + 1
        R[:, j] = r / len(s)
    return R


paths = sorted(glob.glob(os.path.join(SHIPPED_DIR, "raptor_gold_*.npz")))
paths += sorted(glob.glob(os.path.join(NEW_DIR, "raptor_gold_*.npz")))
arms, Y, IDS = {}, None, None
for p in paths:
    d = np.load(p, allow_pickle=True)
    name = os.path.basename(p).replace("raptor_gold_", "").replace(".npz", "")
    i = d["ids"].astype(str)
    if IDS is None:
        IDS, Y = i, d["truth"].astype(float)
        arms[name] = d["pred"].astype(float); continue
    if len(i) == len(IDS) and (i == IDS).all():
        arms[name] = d["pred"].astype(float); continue
    if set(i) == set(IDS):
        order = {u: k for k, u in enumerate(i)}
        arms[name] = d["pred"].astype(float)[np.array([order[u] for u in IDS])]

for t in SHIPPED:
    assert t in arms, "shipped arm missing: %s" % t
ship = rankify(arms[SHIPPED[0]]) + rankify(arms[SHIPPED[1]])
SHIP = macro(Y, ship)

# every 224 arm we hold, discovered so new ones join automatically as the chain finishes
A224 = [a for a in arms if a == "raptor_224" or a.startswith("raptor224_")]
A224.sort()
print("gate %d studies | %d 224 arms: %s" % (len(IDS), len(A224), ", ".join(A224)))
print("SHIPPED 384 PAIR = %.4f   <-- the bar\n" % SHIP)
print("224 arms solo:")
for a in A224:
    print("  %-22s %.4f" % (a, macro(Y, arms[a])))

rng = np.random.default_rng(12345)
boots = [rng.integers(0, len(IDS), len(IDS)) for _ in range(B)]
ship_b = np.array([macro(Y[b], ship[b]) for b in boots])

rows = []
for r in range(1, len(A224) + 1):
    for combo in itertools.combinations(A224, r):
        P = sum(rankify(arms[a]) for a in combo)
        s = macro(Y, P)
        cb = np.array([macro(Y[b], P[b]) for b in boots])
        rows.append((s, float((cb > ship_b).mean()), combo))
rows.sort(reverse=True)
print("\n224-only combinations (rank-mean, exactly how the kernel blends):")
print("  n   gold58   delta   win-rate  arms")
for s, wr, combo in rows[:15]:
    print("  %d  %.4f  %+.4f   %5.1f%%   %s" % (len(combo), s, s - SHIP, wr * 100, " + ".join(combo)))
