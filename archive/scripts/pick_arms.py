#!/usr/bin/env python3
"""Which arms to put in a bigger 384 kernel, chosen for DECORRELATION rather than solo score.

The 224 experiment is closed - 0.928 against the champion's 0.941, for four percent of runtime. The
finding it produced is different: DICOM and window preparation are 78% of the run and are paid ONCE
regardless of arm count, so a 384 arm costs about 3 minutes of GPU and six of them fit in 42 of the
60-minute cap. We ship two.

Arm count only pays if the arms disagree. A 384 trio already scored 0.937 against 0.941 because its
two ConvNeXt arms correlate. So this ranks candidate sets by blended gold score, not by solo score,
and reports the correlation structure that decides it.
"""
import glob, os, itertools
import numpy as np

def auc(y, p):
    o = np.argsort(p); y = y[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = np.arange(1, len(y)+1)
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n0*n1)

def score(pred, truth):
    v = [auc(truth[:, c], pred[:, c]) for c in range(truth.shape[1])]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v))

arms = {}
for f in sorted(glob.glob("goldnew/*.npz")):
    n = os.path.basename(f).replace("raptor_gold_", "").replace(".npz", "")
    d = np.load(f, allow_pickle=True)
    arms[n] = (d["pred"].astype(np.float64), d["truth"].astype(np.float64), d["ids"])
ref = list(arms)[0]
ids0 = arms[ref][2]
arms = {k: v for k, v in arms.items() if len(v[2]) == len(ids0) and (v[2] == ids0).all()}
print("arms with aligned gold rows: %d of %d\n" % (len(arms), len(glob.glob('goldnew/*.npz'))))

def rank(p):
    return np.apply_along_axis(lambda c: np.argsort(np.argsort(c)) / (len(c)-1), 0, p)

solo = {k: score(v[0], v[1]) for k, v in arms.items()}
res = {k: ("224" if "224" in k else "384?") for k in arms}
print("arm                     solo gold   res")
for k, s in sorted(solo.items(), key=lambda kv: -kv[1]):
    print("  %-22s %.4f     %s" % (k, s, res[k]))

names = [k for k in arms if "224" not in k]
R = {k: rank(arms[k][0]) for k in names}
truth = arms[names[0]][1]
print("\npairwise correlation of ranked predictions (384-family arms only):")
print("       " + " ".join("%6s" % n[-6:] for n in names))
for a in names:
    row = " ".join("%6.2f" % np.corrcoef(R[a].ravel(), R[b].ravel())[0,1] for b in names)
    print("  %-5s %s" % (a[-5:], row))

def blend(sel):
    return score(sum(R[k] for k in sel) / len(sel), truth)
print("\nbest set of each size, by blended gold (exhaustive over the 384-family arms):")
for n in range(1, min(7, len(names)+1)):
    best = max(itertools.combinations(names, n), key=blend)
    print("  %d arms  %.4f   %s" % (n, blend(best), " + ".join(best)))
