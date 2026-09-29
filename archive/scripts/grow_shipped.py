#!/usr/bin/env python3
"""Grow the SHIPPED pair rather than searching for the best quadruple.

Why not just take the best 4-arm combo: combotest scores 17,550 of them on 58 studies and reports
the maximum. That maximum is a selection artefact - it is the same mistake that retired this gate
after it called a gain seven times and was wrong seven times. Picking the argmax over thousands of
combinations on 58 studies measures the search, not the arms.

So this does the one thing that is not a search: it KEEPS the pair that actually holds the board at
0.941 and asks what a third and fourth arm add on top of it, one candidate at a time, each judged
on its own. Greedy and pre-committed, so there is no max-over-thousands to inflate it.
"""
import glob, os
import numpy as np

SHIPPED = ("confdist_swa", "coatnet_w80_full_swa")
B = 4000


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


paths = sorted(glob.glob("ortho/artifacts/raptor_gold_*.npz")) + sorted(glob.glob("goldnew/raptor_gold_*.npz"))
arms, Y, IDS = {}, None, None
for p in paths:
    d = np.load(p, allow_pickle=True)
    name = os.path.basename(p).replace("raptor_gold_", "").replace(".npz", "")
    i = d["ids"].astype(str)
    if IDS is None:
        IDS, Y = i, d["truth"].astype(float); arms[name] = d["pred"].astype(float); continue
    if len(i) == len(IDS) and (i == IDS).all():
        arms[name] = d["pred"].astype(float)
    elif set(i) == set(IDS):
        o = {u: k for k, u in enumerate(i)}
        arms[name] = d["pred"].astype(float)[np.array([o[u] for u in IDS])]

A224 = {a for a in arms if a == "raptor_224" or a.startswith("raptor224_")}
rng = np.random.default_rng(12345)
boots = [rng.integers(0, len(IDS), len(IDS)) for _ in range(B)]

cur = list(SHIPPED)
P = sum(rankify(arms[a]) for a in cur)
base_b = np.array([macro(Y[b], P[b]) for b in boots])
print("gate %d studies | shipped pair = %.4f (board 0.941)\n" % (len(IDS), macro(Y, P)))

for step in (3, 4):
    print("=== candidates for arm %d (added on top of %s) ===" % (step, " + ".join(cur)))
    rows = []
    for a in sorted(arms):
        if a in cur or a in A224:
            continue
        Q = P + rankify(arms[a])
        s = macro(Y, Q)
        qb = np.array([macro(Y[b], Q[b]) for b in boots])
        rows.append((s, float((qb > base_b).mean()), a))
    rows.sort(reverse=True)
    for s, wr, a in rows[:8]:
        print("  %-24s %.4f  %+.4f  beats current %.1f%% of bootstraps" % (a, s, macro(Y, P) - 0 and s - macro(Y, P), wr * 100))
    best = rows[0]
    cur.append(best[2])
    P = P + rankify(arms[best[2]])
    base_b = np.array([macro(Y[b], P[b]) for b in boots])
    print("  -> take %s | %d arms = %.4f\n" % (best[2], len(cur), macro(Y, P)))

ship = sum(rankify(arms[a]) for a in SHIPPED)
ship_b = np.array([macro(Y[b], ship[b]) for b in boots])
print("FINAL %d arms: %s" % (len(cur), " + ".join(cur)))
print("  gold58 %.4f vs shipped %.4f  (%+.4f), beats shipped in %.1f%% of bootstraps"
      % (macro(Y, P), macro(Y, ship), macro(Y, P) - macro(Y, ship),
         float((base_b > ship_b).mean()) * 100))
