#!/usr/bin/env python3
"""Score every public label source against our 58 gold studies, then every rank-blend with ours."""
import glob, json, os, sys
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Effusion','Synovitis',"Baker's",'Contusion','Fracture']

tr = pd.read_csv("train.csv", encoding="latin-1")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = tr[tr[LAB].notna().all(axis=1)].copy()
GIDS = gold["StudyInstanceUID"].tolist()
Y = (gold[LAB].values >= 1).astype(int)
print("gold studies %d" % len(GIDS))


def macro(mat):
    """mat: (n_gold, 12) scores aligned to GIDS. NaN columns are skipped, as in the metric."""
    aucs = []
    for j in range(12):
        y, s = Y[:, j], mat[:, j]
        ok = ~np.isnan(s)
        if len(set(y[ok].tolist())) < 2 or ok.sum() < 10:
            aucs.append(np.nan); continue
        aucs.append(roc_auc_score(y[ok], s[ok]))
    return float(np.nanmean(aucs)), aucs


def load_csv(path):
    """Return (n_gold,12) aligned matrix, or None if the file is not a usable label table."""
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    idc = next((c for c in d.columns if "studyinstanceuid" in c.lower().replace("_", "")), None)
    if idc is None:
        return None
    d[idc] = d[idc].astype(str)
    cols = {}
    for f in LAB:
        cand = [c for c in d.columns
                if c.lower().replace("_", " ").replace("'", "").strip() ==
                   f.lower().replace("_", " ").replace("'", "").strip()]
        if not cand:
            key = f.lower().replace(" ", "").replace("'", "")
            cand = [c for c in d.columns
                    if c.lower().replace("_", "").replace(" ", "").replace("'", "") == key]
        if not cand:
            return None
        cols[f] = cand[0]
    d = d.drop_duplicates(subset=[idc]).set_index(idc)
    if not set(GIDS) <= set(d.index):
        return None
    m = d.loc[GIDS, [cols[f] for f in LAB]].apply(pd.to_numeric, errors="coerce").values
    return m.astype(float)


# ours: E[s] over the severity distribution
conf = json.load(open("phase2/labels_v2/conf_all.json"))
ours = np.array([[sum(k * p for k, p in enumerate(conf[u][j])) for j in range(12)]
                 for u in GIDS], dtype=float)

srcs = {"OURS_E[s]": ours}
for p in sorted(glob.glob("publabels/*/*.csv")):
    m = load_csv(p)
    if m is None or np.isnan(m).all():
        continue
    srcs[os.path.relpath(p, "publabels")] = m

print()
print("%-58s %8s" % ("source", "macro"))
print("-" * 68)
scored, leaks = {}, []
for k, m in sorted(srcs.items()):
    s, _ = macro(m)
    # LEAK GUARD: a file that just carries the 58 gold rows scores ~1.0; detect by agreement with gold
    agree = np.nanmean((np.nan_to_num(m, nan=-1) >= 0.5).astype(int) == Y)
    leak = s >= 0.995 or agree >= 0.99
    scored[k] = -1.0 if leak else s
    if leak:
        leaks.append(k)
    print("%-58s %8.4f %s" % (k, s, "<- LEAK: reproduces gold, excluded" if leak else ""))
if leaks:
    print()
    print("excluded %d leaking source(s); every blend containing one was contaminated" % len(leaks))

# keep ours plus the best-scoring variant of each distinct publisher
best_per_pub = {}
for k, s in scored.items():
    if k == "OURS_E[s]":
        continue
    if s < 0:
        continue
    pub = k.split(os.sep)[0]
    if pub not in best_per_pub or s > scored[best_per_pub[pub]]:
        best_per_pub[pub] = k
pool = ["OURS_E[s]"] + sorted(best_per_pub.values())
print()
print("blend pool (ours + best file per publisher): %s" % ", ".join(pool))


def rk(m):
    out = np.full_like(m, np.nan, dtype=float)
    for j in range(12):
        c = m[:, j]
        ok = ~np.isnan(c)
        if ok.sum() > 1:
            out[ok, j] = rankdata(c[ok]) / ok.sum()
    return out


R = {k: rk(srcs[k]) for k in pool}
print()
print("%-58s %8s %8s" % ("rank-mean blend", "macro", "vs ours"))
print("-" * 78)
base = scored["OURS_E[s]"]
res = []
for r in range(1, len(pool) + 1):
    for combo in combinations(pool, r):
        if "OURS_E[s]" not in combo:
            continue
        m = np.nanmean(np.stack([R[k] for k in combo]), axis=0)
        s, per = macro(m)
        res.append((s, combo, per))
for s, combo, per in sorted(res, reverse=True)[:12]:
    print("%-58s %8.4f %+8.4f" % (" + ".join(c.split(os.sep)[0][:26] for c in combo), s, s - base))

best = max(res)
print()
print("BEST %.4f (%+0.4f over ours alone) = %s" % (best[0], best[0] - base, " + ".join(best[1])))
print("per-finding:")
for f, v in zip(LAB, best[2]):
    print("   %-18s %.4f" % (f, v))
json.dump({"best_macro": best[0], "base": base, "combo": list(best[1])},
          open("_blend_labels.json", "w"), indent=1)
