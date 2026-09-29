#!/usr/bin/env python3
"""Put the improved ORDERING onto the champion label set's own distribution."""
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Effusion','Synovitis',"Baker's",'Contusion','Fracture']

cd = pd.read_parquet("labels_confdist.parquet")
cd["StudyInstanceUID"] = cd["StudyInstanceUID"].astype(str)
new = pd.read_parquet("labels_blendimp.parquet")
new["StudyInstanceUID"] = new["StudyInstanceUID"].astype(str)

ids = cd["StudyInstanceUID"].tolist()
src = new.set_index("StudyInstanceUID").reindex(ids)[LAB].values
tgt = cd[LAB].values
assert not np.isnan(src).any(), "blend does not cover every confdist study"

out = np.empty_like(src)
for j in range(12):
    order = np.sort(tgt[:, j])
    idx = np.clip((rankdata(src[:, j], method="ordinal") - 1).astype(int), 0, len(ids) - 1)
    out[:, j] = order[idx]

df = pd.DataFrame(out, columns=LAB)
df.insert(0, "StudyInstanceUID", ids)
df.to_parquet("labels_blendimp_cd.parquet", index=False)

# proof the mapping did what it claims
print("%-18s %12s %12s %12s" % ("finding", "confdist mean", "new mean", "rank corr vs blend"))
for j, f in enumerate(LAB):
    rc = np.corrcoef(rankdata(out[:, j]), rankdata(src[:, j]))[0, 1]
    print("%-18s %12.4f %12.4f %12.4f" % (f, tgt[:, j].mean(), out[:, j].mean(), rc))
print()
print("range: confdist [%.3f, %.3f]  new [%.3f, %.3f]"
      % (tgt.min(), tgt.max(), out.min(), out.max()))
print("rows %d (gold-58 excluded, same as confdist)" % len(df))

# rank correlation of the new order vs confdist per finding
r = [np.corrcoef(rankdata(out[:, j]), rankdata(tgt[:, j]))[0, 1] for j in range(12)]
print("rank correlation to confdist per finding: " + " ".join("%.2f" % x for x in r))
print("mean %.3f - the lower this is, the more the board read is actually testing something" % np.mean(r))
