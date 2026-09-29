#!/usr/bin/env python3
"""Score COMBINATIONS of 2, 3 and 4 arms on the 58-study gate.

Why this is newly worth doing: the old kernel re-read and re-preprocessed every study PER ARM, so a
3rd arm cost +50% wall clock and was rejected. The 1-pass refactor now reads each study ONCE and
runs every arm on the same prepared tensor, and the two T4s run arms concurrently. Reading is most
of the 29 minutes and is shared, so extra arms now cost only a GPU forward pass.
"""
import glob, itertools, os, sys
import numpy as np
from pairtest import auc, macro, rankify, load, SHIPPED

paths = sorted(glob.glob("ortho/artifacts/raptor_gold_*.npz")) + sorted(glob.glob("goldnew/raptor_gold_*.npz"))
arms, Y, ids = load(paths)
R = {k: rankify(v) for k, v in arms.items()}
base = 0.5 * (R[SHIPPED[0]] + R[SHIPPED[1]])
bar = macro(Y, base)
print("gate %d studies | %d arms | SHIPPED PAIR = %.4f\n" % (Y.shape[0], len(arms), bar))

rng = np.random.default_rng(12345)
n = Y.shape[0]
boots = [rng.integers(0, n, n) for _ in range(2000)]
names = sorted(R)

for k in (2, 3, 4):
    rows = []
    for c in itertools.combinations(names, k):
        b = sum(R[x] for x in c) / len(c)
        rows.append((macro(Y, b), c, b))
    rows.sort(key=lambda t: -t[0])
    print("=== BEST %d-ARM COMBOS (%d evaluated) ===" % (k, len(rows)))
    for s, c, b in rows[:6]:
        wins = sum(1 for bt in boots if macro(Y, b, bt) > macro(Y, base, bt))
        wr = wins / len(boots)
        flag = "  <== BEATS BAR @>=85%" if (s > bar and wr >= 0.85) else ""
        print("  %.4f  %+.4f  %5.1f%%  %s%s" % (s, s - bar, 100 * wr, " + ".join(c), flag))
    print()
