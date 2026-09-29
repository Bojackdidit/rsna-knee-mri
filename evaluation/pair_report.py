#!/usr/bin/env python3
"""Score every arm and every pair on the COMMON held-out studies. Usage: pair_report.py [dirs...]

Only arms that HELD OUT the fold may appear here. The champion arms (`champ_*`) trained on all 4,349
studies with no holdout, so the 870 gate studies were in their training data and their 0.939 is
memorisation - they are excluded by name, not by judgement.

Pairs are what ship (one arm per GPU, rank-mean blended), and the pair is not predicted by the solo
scores: cross-family pairs beat same-family ones because the outputs decorrelate (CoAtNet vs effv2s
0.8875, CoAtNet vs CoAtNet 0.9357).

The one anchor to the board: eff384_a + eff384_b scored gate 0.9164 -> BOARD 0.918. It is a single
point, so it gives a direction, not a calibration.
"""
import numpy as np, os, sys, itertools
from sklearn.metrics import roc_auc_score

# --ids FILE restricts scoring to an explicit study list. Needed because arms trained with
# --reclaim hold out only part of the fold: E0 (reclaim 0.7) trained on 520 of the 870 fold-0
# studies, so scoring it on the full fold grades it partly on its own training data and inflates
# it against arms that held the whole fold out. Restricting every arm to the 350 studies E0 never
# saw makes the comparison honest for all of them at once.
_argv = sys.argv[1:]
IDS = ""
if "--ids" in _argv:
    _i = _argv.index("--ids"); IDS = _argv[_i + 1]; del _argv[_i:_i + 2]
DIRS = _argv or ["arms_scott2", "arms_lr", "arms_night", "arms_night_a100"]
EXCLUDE = ("champ_",)
D = {}
for d in DIRS:
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.startswith("raptor_gold_") or not f.endswith(".npz"):
            continue
        tag = f[len("raptor_gold_"):-4]
        if tag in D or any(tag.startswith(x) for x in EXCLUDE):
            continue
        z = np.load(os.path.join(d, f), allow_pickle=True)
        if not {"pred", "truth", "ids"} <= set(z.keys()):
            continue
        D[tag] = (z["ids"].astype(str), z["pred"], z["truth"])
if len(D) < 2:
    sys.exit("need at least two arms with gate dumps")
common = sorted(set.intersection(*[set(v[0]) for v in D.values()]))
if IDS:
    _keep = set(x.strip() for x in open(IDS).read().split() if x.strip())
    _before = len(common)
    common = [u for u in common if u in _keep]
    print("[ids] %s: %d common -> %d scored" % (os.path.basename(IDS), _before, len(common)))
    if not common:
        sys.exit("no overlap between the arms gate dumps and the id list")
idx = {t: {u: i for i, u in enumerate(D[t][0])} for t in D}
P = lambda t: D[t][1][[idx[t][u] for u in common]]
first = sorted(D)[0]
Y = D[first][2][[idx[first][u] for u in common]]
rank = lambda x: x.argsort(0).argsort(0) / max(1, x.shape[0] - 1)


def sc(ps):
    M = np.mean([rank(p) for p in ps], 0)
    return float(np.mean([roc_auc_score((Y[:, j] >= 0.5).astype(int), M[:, j])
                          for j in range(Y.shape[1])
                          if len(set(((Y[:, j] >= 0.5).astype(int)).tolist())) > 1]))


print("arms %d | common held-out studies %d" % (len(D), len(common)))
print("\nSINGLE:")
for v, t in sorted(((sc([P(t)]), t) for t in D), reverse=True):
    print("   %.4f  %s" % (v, t))
res = sorted(((sc([P(a), P(b)]), a, b) for a, b in itertools.combinations(sorted(D), 2)), reverse=True)
print("\nPAIRS (top 12):")
for v, a, b in res[:12]:
    note = "   <- BOARD 0.918" if {a, b} == {"eff384_a", "eff384_b"} else ""
    print("   %.4f  %-18s + %-18s%s" % (v, a, b, note))
anch = [v for v, a, b in res if {a, b} == {"eff384_a", "eff384_b"}]
if anch:
    print("\nvs the only board-anchored pair (%.4f -> board 0.918):" % anch[0])
    for v, a, b in res[:5]:
        print("   %+.4f  %s + %s" % (v - anch[0], a, b))
