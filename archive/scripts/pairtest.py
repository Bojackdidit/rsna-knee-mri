#!/usr/bin/env python3
"""Test every candidate PAIR against the shipped pair on the byte-identical 58-study gate.

What ships is a PAIR of arms rank-averaged, not a single arm, so a slightly weaker arm can still
win if it is decorrelated from the incumbents. This scores every pair offline for free.

Trust check: run with --selftest and it must reproduce the two numbers already established -
shipped pair 0.9333 and best new pair 0.9317. If it does not, the harness is wrong, not the arms.
"""
import argparse, glob, os, sys
import numpy as np

SHIPPED = ("confdist_swa", "coatnet_w80_full_swa")   # what is on the board today


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


def macro(Y, P, rows=None):
    if rows is not None:
        Y, P = Y[rows], P[rows]
    v = [auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def rankify(P):
    """Column-wise rank transform - this is how the kernel blends arms."""
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


def load(paths):
    arms, truth, ids = {}, None, None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        name = os.path.basename(p).replace("raptor_gold_", "").replace(".npz", "")
        i = d["ids"].astype(str)
        if ids is None:
            ids, truth = i, d["truth"]
        else:
            # the gate must be byte-identical across arms or the comparison is meaningless
            if len(i) != len(ids) or not (i == ids).all():
                order = {u: k for k, u in enumerate(i)}
                if set(i) != set(ids):
                    print("  SKIP %s - different study set (%d vs %d)" % (name, len(i), len(ids)))
                    continue
                idx = np.array([order[u] for u in ids])
                arms[name] = d["pred"][idx]
                continue
        arms[name] = d["pred"]
    return arms, truth, ids


ap = argparse.ArgumentParser()
ap.add_argument("--new", default="goldnew")
ap.add_argument("--shipped", default="ortho/artifacts")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--only", default=None, help="only report pairs containing this arm")
a = ap.parse_args()

paths = sorted(glob.glob(os.path.join(a.shipped, "raptor_gold_*.npz")))
paths += sorted(glob.glob(os.path.join(a.new, "raptor_gold_*.npz")))
arms, Y, ids = load(paths)
print("gate: %d studies x %d findings | %d arms loaded\n" % (Y.shape[0], Y.shape[1], len(arms)))

R = {k: rankify(v) for k, v in arms.items()}
singles = {k: macro(Y, v) for k, v in arms.items()}
for k in sorted(singles, key=lambda x: -singles[x]):
    print("  %-28s %.4f" % (k, singles[k]))

if SHIPPED[0] not in R or SHIPPED[1] not in R:
    sys.exit("\nFATAL: shipped arms missing - cannot establish the bar")
base = 0.5 * (R[SHIPPED[0]] + R[SHIPPED[1]])
bar = macro(Y, base)
print("\nSHIPPED PAIR %s + %s = %.4f  <-- the bar\n" % (SHIPPED[0], SHIPPED[1], bar))

names = sorted(R)
rng = np.random.default_rng(a.seed)
n = Y.shape[0]
boots = [rng.integers(0, n, n) for _ in range(a.boot)]

rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        p, q = names[i], names[j]
        if a.only and a.only not in (p, q):
            continue
        blend = 0.5 * (R[p] + R[q])
        s = macro(Y, blend)
        rows.append((s, p, q, blend))
rows.sort(key=lambda t: -t[0])

print("TOP PAIRS (rank-mean, same as the kernel):")
print("  %-52s %8s %8s %s" % ("pair", "gold58", "delta", "boot win-rate vs shipped"))
for s, p, q, blend in rows[:15]:
    wins = 0
    for b in boots:
        if macro(Y, blend, b) > macro(Y, base, b):
            wins += 1
    wr = wins / len(boots)
    flag = "  <== BEATS BAR" if (s > bar and wr >= 0.85) else ""
    print("  %-52s %.4f  %+.4f   %5.1f%%%s" % (p + " + " + q, s, s - bar, 100 * wr, flag))
