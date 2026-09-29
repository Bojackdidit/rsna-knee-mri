#!/usr/bin/env python3
"""Measure what SILENCE means, per finding, on our own corpus and our own gold studies."""
import json, re, sys, unicodedata
import numpy as np, pandas as pd

sys.path.insert(0, "phase2")
from label_extract import ANAT, LAB, norm   # noqa: E402

tr = pd.read_csv("train.csv", encoding="latin-1")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
conf = json.load(open("phase2/labels_v2/conf_all.json"))

rep = dict(zip(tr["StudyInstanceUID"], tr["Report"].fillna("")))
ment = {}
for uid, txt in rep.items():
    t = norm(txt)
    ment[uid] = {f: bool(re.search(ANAT[f], t)) for f in LAB}

gold = tr[tr[LAB].notna().all(axis=1)].copy()
print("corpus %d studies | gold %d studies" % (len(tr), len(gold)))
print()
print("%-18s %8s %8s %8s %8s %6s" % ("finding", "silent%", "P+|sil", "P+|spk", "n_sil", "lift"))
print("-" * 62)
rows = []
for j, f in enumerate(LAB):
    sil_all = np.mean([not ment[u][f] for u in tr["StudyInstanceUID"] if u in ment])
    g = gold["StudyInstanceUID"].tolist()
    y = (gold[f].values >= 1).astype(int)
    s = np.array([not ment[u][f] for u in g])
    p_sil = y[s].mean() if s.any() else float("nan")
    p_spk = y[~s].mean() if (~s).any() else float("nan")
    rows.append((f, sil_all, p_sil, p_spk, int(s.sum())))
    print("%-18s %7.1f%% %8.2f %8.2f %8d %6s"
          % (f, 100 * sil_all, p_sil, p_spk, s.sum(),
             "%.2f" % (p_sil / p_spk) if p_spk and p_spk == p_spk and p_spk > 0 else "-"))
print()
print("Reading: a HIGH P+|sil means silence does NOT mean absent, so a confident zero there is a")
print("false negative and the cell is worth imputing. A near-zero P+|sil means silence IS the label")
print("and imputing would destroy real information.")
json.dump([{"finding": r[0], "silent_frac": r[1], "p_pos_silent": r[2],
            "p_pos_spoken": r[3], "n_gold_silent": r[4]} for r in rows],
          open("_silence_table.json", "w"), indent=1)
