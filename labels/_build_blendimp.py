#!/usr/bin/env python3
"""Build the candidate training label set: rank-blend of two extractions plus a targeted silence repair."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
REPAIR_PUB = ['Synovitis', 'PF OA', 'Fracture']      # stevenleehans' published set
REPAIR = ['PF OA', 'Fracture']

tr = pd.read_csv("train.csv", encoding="latin-1")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
IDS = tr["StudyInstanceUID"].tolist()
pos = {u: i for i, u in enumerate(IDS)}
gold = tr[tr[LAB].notna().all(axis=1)]
GIDS = gold["StudyInstanceUID"].tolist()
gi = np.array([pos[u] for u in GIDS])
Y = (gold[LAB].values >= 1).astype(int)

conf = json.load(open("phase2/labels_v2/conf_all.json"))
ours = np.array([[sum(k * p for k, p in enumerate(conf[u][j])) for j in range(12)] for u in IDS])

sl = pd.read_csv("publabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", low_memory=False)
sl["StudyInstanceUID"] = sl["StudyInstanceUID"].astype(str)
sl = sl.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
theirs = sl.reindex(IDS)[LAB].apply(pd.to_numeric, errors="coerce").values

pk = pd.read_csv("publabels/rsna-knee-llm-labels/report_labels_v2.csv", low_memory=False)
pk["StudyInstanceUID"] = pk["StudyInstanceUID"].astype(str)
pk = pk.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(IDS)
UNK = np.stack([(pk[f + "__verdict"] == "UNK").values for f in LAB], 1)   # missing -> False
print("sources aligned on %d studies | gold %d | UNK cells %.1f%%"
      % (len(IDS), len(GIDS), 100 * UNK.mean()))


def rk(m):
    out = np.full(m.shape, np.nan)
    for j in range(m.shape[1]):
        c = m[:, j]; ok = ~np.isnan(c)
        out[ok, j] = rankdata(c[ok]) / ok.sum()
    return out


def macro(M, name):
    s = M[gi]
    a = []
    for j in range(12):
        ok = ~np.isnan(s[:, j])
        a.append(roc_auc_score(Y[ok, j], s[ok, j]) if len(set(Y[ok, j].tolist())) > 1 else np.nan)
    print("%-34s %.4f" % (name, np.nanmean(a)))
    return float(np.nanmean(a)), a


Ro, Rt = rk(ours), rk(theirs)
blend = np.nanmean(np.stack([Ro, Rt]), 0)
m_ours, _ = macro(rk(ours), "ours E[s] alone")
m_bl, a_bl = macro(blend, "blend (ours + stevenleehans v4)")


def impute(base, cols):
    out = base.copy()
    fitrows = np.ones(len(IDS), bool)
    fitrows[gi] = False                       # gold never enters the fit
    for f in cols:
        j = LAB.index(f)
        others = [k for k in range(12) if k != j]
        X, y = base[:, others], base[:, j]
        ok = fitrows & ~UNK[:, j] & ~np.isnan(y) & ~np.isnan(X).any(1)
        if ok.sum() < 200:
            print("  skip %s (only %d clean fit rows)" % (f, ok.sum())); continue
        r = Ridge(alpha=1.0).fit(X[ok], y[ok])
        tgt = UNK[:, j] & ~np.isnan(X).any(1)
        out[tgt, j] = r.predict(X[tgt])
        print("  %-16s repaired %5d UNK cells (%.1f%% of column)"
              % (f, tgt.sum(), 100 * tgt.mean()))
    return out


print()
print("targeted repair (PF OA + Fracture):")
imp3 = impute(blend, REPAIR)
print("published-set repair (adds Synovitis):")
imppub = impute(blend, REPAIR_PUB)
print("blanket control:")
impall = impute(blend, LAB)
print()
m3, a3 = macro(imp3, "blend + repair(PF OA, Fracture)")
mpub, _ = macro(imppub, "blend + repair(published 3)")
mall, _ = macro(impall, "blend + repair(all 12) CONTROL")

print()
print("%-18s %8s %8s %8s" % ("finding", "blend", "repaired", "delta"))
for f, b, r in zip(LAB, a_bl, a3):
    mark = " *" if f in REPAIR else ""
    print("%-18s %8.4f %8.4f %+8.4f%s" % (f, b, r, r - b, mark))

for name, M in (("blendimp", imp3), ("blend", blend), ("blendimp_pub", imppub)):
    o = pd.DataFrame(M, columns=LAB)
    o.insert(0, "StudyInstanceUID", IDS)
    o.to_parquet("labels_%s.parquet" % name, index=False)
tag = "blendimp"
print()
print("wrote labels_%s.parquet  (gold-58 macro %.4f vs %.4f for ours alone)"
      % (tag, max(m3, m_bl), m_ours))
print("58 studies cannot resolve under ~0.02 and have mispredicted the board 8+ times.")
print("This ranks a CANDIDATE. The board decides.")

# calibration: quantile-map each blended column onto our own E[s] marginals so scale/prevalence match the loss
o = np.sort(ours, axis=0)
cal = np.empty_like(imp3)
for j in range(12):
    r = imp3[:, j]
    idx = np.clip((rankdata(r) - 1).astype(int), 0, len(IDS) - 1)
    cal[:, j] = o[idx, j]
m_cal, _ = macro(cal, "blend+repair CALIBRATED")
print("prevalence check (mean label value):")
print("   ours      " + " ".join("%.2f" % v for v in ours.mean(0)))
print("   rank      " + " ".join("%.2f" % v for v in imp3.mean(0)))
print("   calibrated" + " ".join("%.2f" % v for v in cal.mean(0)))
oc = pd.DataFrame(cal, columns=LAB)
oc.insert(0, "StudyInstanceUID", IDS)
oc.to_parquet("labels_blendimp_cal.parquet", index=False)
print("wrote labels_blendimp_cal.parquet")
