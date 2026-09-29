#!/usr/bin/env python3
"""Do the 224 arms add anything to the 384 arms, ignoring shippability?

The kernel cannot currently run both resolutions - it builds each study's windows once and asserts
one resolution. Lifting that means preparing windows twice per study, which is real work in the hot
path. This asks whether it would be worth doing: take the shipped 384 pair and every 384 arm, and
see whether adding a 224 arm to a mix ever beats the same mix without it.

If a 224 arm never earns its place next to the 384 arms, 224 is a dead end and mixed-resolution
support should not be built.
"""
import glob, itertools, os
import numpy as np

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

A224 = sorted(a for a in arms if a == "raptor_224" or a.startswith("raptor224_"))
A384 = [a for a in arms if a not in A224]
rng = np.random.default_rng(12345)
boots = [rng.integers(0, len(IDS), len(IDS)) for _ in range(B)]

ship = rankify(arms[SHIPPED[0]]) + rankify(arms[SHIPPED[1]])
SHIP = macro(Y, ship)
ship_b = np.array([macro(Y[b], ship[b]) for b in boots])
print("gate %d studies | %d arms at 384, %d at 224" % (len(IDS), len(A384), len(A224)))
print("SHIPPED 384 PAIR = %.4f\n" % SHIP)

# 1. the shipped pair, plus each 224 arm
print("shipped pair + one 224 arm:")
for a in A224:
    P = ship + rankify(arms[a])
    s = macro(Y, P)
    wr = float((np.array([macro(Y[b], P[b]) for b in boots]) > ship_b).mean())
    print("  + %-20s %.4f  %+.4f  win %.1f%%" % (a, s, s - SHIP, wr * 100))

# 2. does a 224 arm ever improve the BEST 3-arm mixes? compare best-with vs best-without
best_wo, best_w = [], []
for combo in itertools.combinations(A384, 3):
    P = sum(rankify(arms[a]) for a in combo)
    best_wo.append((macro(Y, P), combo))
for combo384 in itertools.combinations(A384, 2):
    base = sum(rankify(arms[a]) for a in combo384)
    for a in A224:
        P = base + rankify(arms[a])
        best_w.append((macro(Y, P), combo384 + (a,)))
best_wo.sort(reverse=True); best_w.sort(reverse=True)
print("\nbest 3-arm, 384 only:      %.4f  %s" % (best_wo[0][0], " + ".join(best_wo[0][1])))
print("best 3-arm, includes a 224: %.4f  %s" % (best_w[0][0], " + ".join(best_w[0][1])))
P = sum(rankify(arms[a]) for a in best_w[0][1])
wr = float((np.array([macro(Y[b], P[b]) for b in boots]) > ship_b).mean())
print("   that mixed trio vs shipped pair: %+.4f, win %.1f%%" % (best_w[0][0] - SHIP, wr * 100))
