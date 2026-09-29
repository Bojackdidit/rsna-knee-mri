#!/usr/bin/env python3
"""Anti-scanner augmentation, judged on the CORRECT instrument for the first time.

--aug_scanner was refuted once (SCANNER_ARM/SCANNER_E24) on gold-58 - a scanner-CONTAMINATED gate
that structurally rewards the very scanner shortcut aug_scanner removes, so it was always going to
look worse there. This compares the anti-scanner arm (SCAN_GEN_F0) against the identical-recipe
control (LR96_CTRL) on the SAME 870 held-out-SCANNER studies of scanner-grouped fold-0 - the honest
cross-scanner generalization gate, which aug_scanner has never been measured on.

Decision: does aug_scanner lift held-out-scanner macro-AUC beyond the 0.0028 OOF noise floor, and
does it shrink the per-scanner AUC spread (the mechanism)? If yes -> board-test. If flat/down on the
right instrument too -> aug_scanner is genuinely dead, closed properly this time.
"""
import numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def macro(P, Y, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    a = [roc_auc_score((Y[idx, j] >= 0.5).astype(int), P[idx, j])
         for j in range(Y.shape[1]) if len(set((Y[idx, j] >= 0.5).astype(int).tolist())) > 1]
    return float(np.mean(a)) if a else float("nan")


ctrl = np.load("oof/oof_LR96_CTRL.npz", allow_pickle=True)
scan = np.load("oof/oof_SCAN_GEN_F0.npz", allow_pickle=True)
c_ids = np.asarray(ctrl["ids"], dtype=object).astype(str)
s_ids = np.asarray(scan["ids"], dtype=object).astype(str)
assert set(c_ids) == set(s_ids), "fold-0 study sets differ between control and aug_scanner arm"
poss = {s: i for i, s in enumerate(s_ids)}
order = [poss[s] for s in c_ids]
Pc, Y = ctrl["pred"].astype(float), ctrl["truth"].astype(float)
Ps = scan["pred"].astype(float)[order]

mc, ms = macro(Pc, Y), macro(Ps, Y)
n = len(Y)
print("held-out-SCANNER fold-0 studies:", n)
print("  LR96_CTRL (no aug_scanner) macro : %.4f" % mc)
print("  SCAN_GEN_F0 (aug_scanner)  macro : %.4f   (%+.4f vs control)" % (ms, ms - mc))

rng = np.random.default_rng(0)
d = np.empty(2000)
for i in range(2000):
    idx = rng.integers(0, n, n)
    d[i] = macro(Ps, Y, idx) - macro(Pc, Y, idx)
print("  aug_scanner vs control : mean %+.4f, win-rate %.1f%% (2000 paired bootstraps)"
      % (d.mean(), 100 * (d > 0).mean()))
print("  noise floor for a real delta on this fold ~0.0028 (bootstrap sd of the OOF instrument)")

# --- mechanism: does aug_scanner shrink the per-scanner AUC spread? ---
fp = pd.read_csv("study_fingerprint.csv")
id2fp = dict(zip(fp["StudyInstanceUID"].astype(str), fp["fp"].astype(str)))
fps = np.array([id2fp.get(s, "UNK") for s in c_ids])
by = {}
for f in set(fps.tolist()):
    m = fps == f
    if m.sum() >= 30:  # enough studies for a stable per-scanner macro
        by[f] = (macro(Pc, Y, np.where(m)[0]), macro(Ps, Y, np.where(m)[0]), int(m.sum()))
if by:
    cs = np.array([v[0] for v in by.values()]); ss = np.array([v[1] for v in by.values()])
    print("\nper-scanner macro spread (scanners with >=30 held-out studies, n=%d):" % len(by))
    print("  control  spread %.4f  (%.4f - %.4f)" % (cs.max() - cs.min(), cs.min(), cs.max()))
    print("  aug_scan spread %.4f  (%.4f - %.4f)  <- smaller = more scanner-invariant"
          % (ss.max() - ss.min(), ss.min(), ss.max()))

print("\nper-finding (control / aug_scanner / delta):")
for j, name in enumerate(LAB):
    y = (Y[:, j] >= 0.5).astype(int)
    if len(set(y.tolist())) < 2:
        continue
    ac, as_ = roc_auc_score(y, Pc[:, j]), roc_auc_score(y, Ps[:, j])
    print("  %-18s %.4f / %.4f  %+.4f" % (name, ac, as_, as_ - ac))
