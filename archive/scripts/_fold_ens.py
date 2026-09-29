"""
Score a fold ensemble as a single virtual arm, then rank it as a partner.

The fold lever is the only competitor-named number still standing after labels and architecture were
both refuted for us: Scott went 0.938 single fold -> 0.947 five folds, and the #1 team put folds at
+0.009 independently. But a fold model on its own is NOT the lever - FOLD0 alone reads 0.9168 and
pairs at 0.9297, below the champion. The ensemble is the thing, so it has to be scored as one.

Each fold sees a different 80% of the studies, which is decorrelation from DATA rather than from
architecture - and architecture is exhausted after four families.
"""
import glob, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

CHAMP = "SCRATCH_AUG2"
folds = sorted(f for f in glob.glob("gold_oof/raptor_gold_FOLD*.npz"))
if not folds:
    raise SystemExit("no fold gold files yet")
arms = {os.path.basename(f)[12:-4]: np.load(f, allow_pickle=True) for f in glob.glob("gold_oof/raptor_gold_*.npz")}
ids = arms[CHAMP]["ids"]
for k, v in list(arms.items()):
    if not np.array_equal(v["ids"], ids):
        arms.pop(k)
truth = arms[CHAMP]["truth"]; n, nf = truth.shape
Y = (truth >= 0.5).astype(int)
R = {k: np.stack([rankdata(v["pred"][:, j]) / n for j in range(nf)], 1) for k, v in arms.items()}
macro = lambda p: float(np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(nf) if 0 < Y[:, j].sum() < n]))
rho = lambda a, b: float(np.mean([spearmanr(a[:, j], b[:, j]).statistic for j in range(nf)]))

names = [os.path.basename(f)[12:-4] for f in folds]
names = [x for x in names if x in R]
print(f"{len(names)} fold model(s): {', '.join(names)}\n")
for k in names:
    print(f"  {k:12s} solo {macro(arms[k]['pred']):.4f}   rho vs champion {rho(R[k], R[CHAMP]):.4f}")

ens = sum(R[k] for k in names) / len(names)
print(f"\n  fold ensemble of {len(names)}   solo {macro(ens):.4f}   rho vs champion {rho(ens, R[CHAMP]):.4f}")
print(f"  ensemble + champion pair          {macro((ens + R[CHAMP]) / 2):.4f}")
print(f"  shipped champion pair             {macro((R['CONVNEXT_E24'] + R[CHAMP]) / 2):.4f}")
print(f"  ensemble + ConvNeXt + champion    {macro((ens + R['CONVNEXT_E24'] + R[CHAMP]) / 3):.4f}"
      f"   (trios have lost twice on the board)")
