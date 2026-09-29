#!/usr/bin/env python3
"""Out-of-fold validation over all 4,349 labelled studies - the instrument we have never had.

Why this exists. Every offline call this week was made on 58 gold studies, and gold-58 predicted
the WRONG SIGN on two board reads in one day (per-finding routing, three-family trio). The 870
grouped fold is no better, because the shipped arms trained on all 4,349 with no holdout and so
have memorised it. So we have been choosing blind.

Fold k holds out its own 870 studies. Train all five and every one of the 4,349 studies has a
prediction from a model that never saw it. Concatenated, that is a validation set 75x gold-58,
honest by construction, and it is what a serious team means by a validation pipeline.

    python _oof_validate.py                 # score the OOF assembly
    python _oof_validate.py --per-finding   # add the per-finding breakdown

Caveat kept in view: the soft labels are our own, so this measures agreement with them, not with
radiologist truth ([[knee-gate-grades-own-labels]]). Size fixes the variance problem, not the
label-bias one. Gold-58 is still reported alongside as the only radiologist-labelled check.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FOLD_ARMS = {0: "LR96_CTRL", 1: "LR96_F1", 2: "LR96_F2", 3: "LR96_F3", 4: "LR96_F4"}


def macro(P, Y, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    a = [roc_auc_score((Y[idx, j] >= 0.5).astype(int), P[idx, j])
         for j in range(Y.shape[1]) if len(set((Y[idx, j] >= 0.5).astype(int).tolist())) > 1]
    return float(np.mean(a)) if a else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="oof")
    ap.add_argument("--per-finding", action="store_true")
    a = ap.parse_args()

    parts, missing = [], []
    for f, tag in sorted(FOLD_ARMS.items()):
        p = os.path.join(a.dir, "oof_%s.npz" % tag)
        if not os.path.exists(p):
            missing.append("fold %d (%s)" % (f, tag))
            continue
        d = np.load(p, allow_pickle=True)
        parts.append((f, np.asarray(d["ids"], dtype=object).astype(str), d["pred"], d["truth"]))
        print("  fold %d  %-10s %4d studies" % (f, tag, len(d["ids"])))

    if missing:
        print("\nMISSING: %s" % ", ".join(missing))
        print("Each fold arm must be scored on ITS OWN held-out studies - scoring a fold-k arm on")
        print("fold 0 grades it on its own training data (that is what made E0 read 0.9536 against")
        print("a true 0.8961). Nothing is reported until the assembly is complete.")
        if not parts:
            return

    ids = np.concatenate([p[1] for p in parts])
    P = np.concatenate([p[2] for p in parts])
    Y = np.concatenate([p[3] for p in parts])
    dup = len(ids) - len(set(ids.tolist()))
    print("\nOOF assembly: %d studies, %d duplicated ids (must be 0)" % (len(ids), dup))
    assert dup == 0, "folds overlap - the partition is not disjoint"

    print("OOF macro-AUC over %d studies: %.4f" % (len(ids), macro(P, Y)))
    n = len(ids)
    rng = np.random.default_rng(0)
    boot = [macro(P, Y, rng.integers(0, n, n)) for _ in range(200)]
    print("  bootstrap sd %.4f  (gold-58's sd on the same statistic is ~0.02)" % float(np.std(boot)))

    if a.per_finding:
        print("\nper-finding, %d studies:" % len(ids))
        for j, name in enumerate(LAB):
            y = (Y[:, j] >= 0.5).astype(int)
            if len(set(y.tolist())) > 1:
                print("  %-18s %.4f   (positives %4d)" % (name, roc_auc_score(y, P[:, j]), int(y.sum())))


if __name__ == "__main__":
    main()
