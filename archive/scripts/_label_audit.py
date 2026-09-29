"""
Survey every public label set against the 58 real annotator rows, with the leak guard first.

Yann Majewski reports that blending his labels with a public dataset was worth **+0.015 of board** -
the largest number anyone in that thread attached to anything. We dismissed public label sets after
three of them turned out to contain the 58 gold rows verbatim and score a perfect 1.0000; that was a
leak test on three files, not a survey, and it is the only lever from the leaders' own accounts that
we have not actually looked at.

Two things matter per file: whether it LEAKS (a set that simply contains the gold rows teaches
nothing and would poison every gate we have), and whether it DIFFERS from ours. A label set that
agrees with ours everywhere cannot add anything; the value is in where an independent reading of the
same report disagrees.
"""
import glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

tr = pd.read_csv("train.csv") if os.path.exists("train.csv") else None
if tr is None:
    raise SystemExit("train.csv not found locally - run this where the real labels live")
gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
print(f"{len(gold)} studies carry real annotator labels\n")

ours = pd.read_parquet("labels_confdist.parquet").set_index("StudyInstanceUID")


def score(df, name):
    cols = [c for c in LAB if c in df.columns]
    if len(cols) < 8:
        print(f"{name:52s} only {len(cols)}/12 findings - skipped")
        return
    idx = gold.index.intersection(df.index)
    if len(idx) < 20:
        print(f"{name:52s} covers only {len(idx)} gold studies - skipped")
        return
    aucs, leak = [], 0
    for c in cols:
        y = (gold.loc[idx, c].values >= 0.5).astype(int)
        p = pd.to_numeric(df.loc[idx, c], errors="coerce").values
        if np.isnan(p).any() or len(set(y)) < 2:
            continue
        a = roc_auc_score(y, p)
        aucs.append(a)
        leak += a > 0.999
    both = ours.index.intersection(df.index)
    common = [c for c in cols if c in ours.columns]
    dis = float(np.mean(np.abs(pd.to_numeric(df.loc[both, common].stack(), errors="coerce").values
                               - ours.loc[both, common].stack().values)))
    flag = "  *** LEAKS ***" if leak >= 6 else ""
    print(f"{name:52s} gold-AUC {np.mean(aucs):.4f}  perfect-cols {leak:2d}/12  "
          f"studies {len(df):5d}  mean |diff| vs ours {dis:.4f}{flag}")


for f in sorted(glob.glob("publabels/*/*.csv")):
    try:
        d = pd.read_csv(f)
    except Exception as e:
        print(f"{f:52s} unreadable: {e}")
        continue
    key = next((c for c in d.columns if "Study" in c or "study" in c), None)
    if key is None:
        print(f"{os.path.relpath(f):52s} no study id column - skipped")
        continue
    score(d.set_index(key), os.path.relpath(f, "publabels"))

print()
score(ours.reset_index().set_index("StudyInstanceUID"), "OURS labels_confdist.parquet")
