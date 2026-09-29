#!/usr/bin/env python3
"""Score every combination of the five 224 arms on the gold set, so the ones worth submitting are
chosen by measurement rather than by which happened to be mentioned in a note.

Arms are averaged by rank, exactly as the shipped kernel does.
"""
import glob, os, itertools
import numpy as np

def auc(y, p):
    o = np.argsort(p); y = y[o]
    n1 = int(y.sum()); n0 = len(y) - n1
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
    if "224" not in n: continue
    d = np.load(f, allow_pickle=True)
    arms[n] = (d["pred"].astype(np.float64), d["truth"].astype(np.float64), d["ids"])
ids0 = list(arms.values())[0][2]
arms = {k: v for k, v in arms.items() if len(v[2]) == len(ids0) and (v[2] == ids0).all()}
truth = list(arms.values())[0][1]
def rank(p): return np.apply_along_axis(lambda c: np.argsort(np.argsort(c))/(len(c)-1), 0, p)
R = {k: rank(v[0]) for k, v in arms.items()}
print("224 arms on the gold set (%d studies):" % len(ids0))
for k in sorted(R, key=lambda k: -score(arms[k][0], truth)):
    print("   %-20s %.4f" % (k, score(arms[k][0], truth)))
names = sorted(R)
print("\nevery combination, best first:")
res = []
for n in range(1, len(names)+1):
    for c in itertools.combinations(names, n):
        res.append((score(sum(R[k] for k in c)/len(c), truth), c))
res.sort(reverse=True)
for s, c in res[:10]:
    print("   %.4f  (%d) %s" % (s, len(c), " + ".join(x.replace("raptor224_","").replace("raptor_224","seed777") for x in c)))
print("\nbest at each size:")
for n in range(1, len(names)+1):
    b = max((r for r in res if len(r[1]) == n))
    print("   %d arms  %.4f   %s" % (n, b[0], " + ".join(b[1])))
