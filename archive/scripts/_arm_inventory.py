#!/usr/bin/env python3
"""Score every arm we already own on ONE common, uncontaminated ruler. Zero GPU, zero training.

Two faults made every previous cross-arm comparison invalid:
  1. each `raptor_gold_*.npz` carries its OWN `truth`, taken from whichever soft-label parquet that
     arm trained on, so arms on different label sets were graded against different targets
     (memory/knee-gate-grades-own-labels);
  2. some dumps are fold-validation sets the arm partly trained on
     (memory/score-arms-only-on-their-own-holdout).
Both are fixed the same way: ignore the stored truth entirely, intersect each dump with the 58
studies that carry REAL annotator labels in train.csv, and score there. No arm trains on those.
"""
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
tr = pd.read_csv("train.csv", encoding="latin-1")
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = tr[tr[LAB].notna().all(axis=1)]
GY = {u: (gold.loc[gold["StudyInstanceUID"] == u, LAB].values[0] >= 1).astype(int)
      for u in gold["StudyInstanceUID"]}
GIDS = set(GY)
print("real-gold studies: %d" % len(GIDS))

rows = []
for p in sorted(glob.glob("**/raptor_gold_*.npz", recursive=True)):
    try:
        z = np.load(p, allow_pickle=True)
        ids = np.array([str(x) for x in z["ids"]])
        pred = np.asarray(z["pred"], dtype=float)
    except Exception:
        continue
    if pred.ndim != 2 or pred.shape[1] != 12 or len(ids) != len(pred):
        continue
    m = np.array([u in GIDS for u in ids])
    if m.sum() < 40:
        continue
    P = pred[m]
    Y = np.stack([GY[u] for u in ids[m]])
    a = []
    for j in range(12):
        if len(set(Y[:, j].tolist())) > 1:
            a.append(roc_auc_score(Y[:, j], P[:, j]))
    rows.append({"file": p, "n_gold": int(m.sum()), "macro": float(np.mean(a)),
                 "tag": os.path.basename(p)[12:-4]})
d = pd.DataFrame(rows).sort_values("macro", ascending=False)
d.to_csv("_arm_inventory.csv", index=False)
print("scored %d dumps on the real gold labels" % len(d))
print()
print(d.head(25).to_string(index=False))
