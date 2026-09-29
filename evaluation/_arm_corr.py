"""
Rank candidate partners for the shipped champion by DECORRELATION, which is what actually decides a
pair on this competition, and only then by solo score.

Measured 2026-09-17 on the three arms whose gold files survived: a same-architecture partner sits at
rank correlation 0.958 and adds +0.0020, a different-architecture partner sits at 0.88-0.90 and adds
+0.0047 to +0.0062. The best pair was built from the WEAKEST solo arm. So solo gold is the wrong
sort key.

Correlation is per finding on ranks, because the blend is a rank-mean and ranks are what gets
averaged. Everything is measured on the 58 gold studies, which is small and grades our own soft
labels, so trust the ORDERING and not the magnitudes.

Usage:
  python _arm_corr.py                       # every npz in gold_oof, champion auto-detected
  python _arm_corr.py --champ SCRATCH_AUG2  # name the champion explicitly
"""
import argparse, glob, itertools, os
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="gold_oof")
ap.add_argument("--champ", default="SCRATCH_AUG2", help="tag of the shipped champion")
ap.add_argument("--pairs", action="store_true", help="also print the full pairwise table")
ap.add_argument("--trios", action="store_true",
                help="score every three-arm blend too; a third arm costs about 21 minutes "
                     "of kernel time and a correlated one has already been shown to dilute")
a = ap.parse_args()

arms = {}
for f in sorted(glob.glob(os.path.join(a.dir, "raptor_gold_*.npz"))):
    tag = os.path.basename(f)[len("raptor_gold_"):-len(".npz")]
    arms[tag] = np.load(f, allow_pickle=True)
if not arms:
    raise SystemExit(f"no gold files in {a.dir}/")

ids = truth = None
for tag, d in list(arms.items()):
    if ids is None:
        ids, truth = d["ids"], d["truth"]
    elif not np.array_equal(ids, d["ids"]):
        print(f"  dropping {tag}: different study order")
        arms.pop(tag)
n, nf = truth.shape
Y = (truth >= 0.5).astype(int)


def macro(p):
    v = [roc_auc_score(Y[:, j], p[:, j]) for j in range(nf) if 0 < Y[:, j].sum() < n]
    return float(np.mean(v))


R = {t: np.stack([rankdata(d["pred"][:, j]) / n for j in range(nf)], 1) for t, d in arms.items()}
solo = {t: macro(arms[t]["pred"]) for t in arms}

print(f"{n} studies, {nf} findings, {len(arms)} arms\n")
if a.champ not in arms:
    a.champ = max(solo, key=solo.get)
    print(f"champion not found, falling back to the best solo arm: {a.champ}\n")

c = R[a.champ]
print(f"partners for {a.champ} (solo {solo[a.champ]:.4f}), sorted by correlation - LOWEST FIRST\n")
print(f"  {'partner':22s} {'solo':>7s} {'rho':>7s} {'pair':>7s} {'vs champ':>9s} {'vs best solo':>13s}")
rows = []
for t in arms:
    if t == a.champ:
        continue
    rho = float(np.mean([spearmanr(c[:, j], R[t][:, j]).statistic for j in range(nf)]))
    pair = macro((c + R[t]) / 2)
    rows.append((rho, t, solo[t], pair))
for rho, t, s, pair in sorted(rows):
    print(f"  {t:22s} {s:7.4f} {rho:7.4f} {pair:7.4f} {pair-solo[a.champ]:+9.4f} "
          f"{pair-max(s, solo[a.champ]):+13.4f}")

if a.pairs:
    print("\nall pairs")
    for x, y in itertools.combinations(sorted(arms), 2):
        rho = float(np.mean([spearmanr(R[x][:, j], R[y][:, j]).statistic for j in range(nf)]))
        print(f"  {x:20s} + {y:20s} rho {rho:.4f}  pair {macro((R[x]+R[y])/2):.4f}")

if a.trios and len(arms) >= 3:
    print("\nall trios - a third arm costs about 21 minutes of kernel time because it shares a card,")
    print("so it has to beat the best PAIR, not the champion. A correlated third member dilutes.")
    best_pair = max(macro((R[x] + R[y]) / 2) for x, y in itertools.combinations(sorted(arms), 2))
    out = [(macro((R[x] + R[y] + R[z]) / 3), x, y, z)
           for x, y, z in itertools.combinations(sorted(arms), 3)]
    for sc, x, y, z in sorted(out, reverse=True):
        print(f"  {x:18s} + {y:18s} + {z:18s} {sc:.4f}  {sc - best_pair:+.4f} vs best pair")
