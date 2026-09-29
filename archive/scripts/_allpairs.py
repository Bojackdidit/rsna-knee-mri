#!/usr/bin/env python3
"""Rank every arm and every PAIR on the 870-study grouped fold, from dumps already on disk.

No GPU. 76 dumps exist with pred/truth/ids saved, which is exactly what they were saved for.

Two traps this guards against:
  - cross-label comparison. The gate scores AGREEMENT WITH ITS LABEL SET, so two arms graded
    against different truth are not comparable. Dumps are grouped by a hash of their truth
    array and only compared inside a group.
  - id order. Dumps are aligned on ids before pairing, not assumed to share row order.

The pair score is the 0.5/0.5 rank-mean the kernel actually ships.
"""
import glob
import hashlib
import os
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def macro(P, Y):
    a = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j])
         for j in range(Y.shape[1]) if len(set((Y[:, j] >= 0.5).astype(int).tolist())) > 1]
    return float(np.mean(a))


def main():
    arms = {}
    for f in glob.glob(os.path.join(".", "**", "raptor_gold_*.npz"), recursive=True):
        try:
            d = np.load(f, allow_pickle=True)
            if "pred" not in d or d["pred"].shape[0] != 870:
                continue
            tag = os.path.basename(f)[12:-4]
            ids = np.asarray(d["ids"], dtype=object).astype(str)
            key = hashlib.md5(np.ascontiguousarray(d["truth"]).tobytes()
                              + "".join(sorted(ids)).encode()).hexdigest()[:8]
            arms.setdefault((key, tag), (d["pred"], d["truth"], ids, f))
        except Exception:
            continue

    groups = defaultdict(dict)
    for (key, tag), v in arms.items():
        groups[key][tag] = v
    print("truth-set groups on the 870 fold: %s"
          % {k: len(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))})

    key = max(groups, key=lambda k: len(groups[k]))
    g = groups[key]
    print("\n=== largest comparable group: %s | %d arms ===" % (key, len(g)))

    ref_ids = None
    solo = {}
    P = {}
    for tag, (p, y, ids, f) in sorted(g.items()):
        order = np.argsort(ids)
        if ref_ids is None:
            ref_ids, Y = ids[order], y[order]
        P[tag] = p[order]
        solo[tag] = macro(P[tag], Y)

    print("\n--- solo, best 14 ---")
    for tag, v in sorted(solo.items(), key=lambda kv: -kv[1])[:14]:
        print("  %-26s %.4f" % (tag, v))

    R = {t: np.column_stack([rankdata(P[t][:, j]) for j in range(P[t].shape[1])]) for t in P}
    tags = sorted(P)
    pairs = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            v = macro(0.5 * R[a] + 0.5 * R[b], Y)
            rho = float(np.mean([np.corrcoef(R[a][:, c], R[b][:, c])[0, 1]
                                 for c in range(R[a].shape[1])]))
            pairs.append((v, a, b, rho, max(solo[a], solo[b])))
    pairs.sort(reverse=True)

    print("\n--- best 14 PAIRS (rank-mean 0.5/0.5) ---")
    print("  %-40s %8s %8s %7s %7s" % ("pair", "pair", "bestsolo", "gain", "rho"))
    for v, a, b, rho, bs in pairs[:14]:
        print("  %-40s %8.4f %8.4f %+7.4f %7.3f" % (a + " + " + b, v, bs, v - bs, rho))

    champs = [p for p in pairs if {"champ_confdist", "champ_w80"} == {p[1], p[2]}]
    if champs:
        v, a, b, rho, bs = champs[0]
        rank = pairs.index(champs[0]) + 1
        print("\n  shipped champion pair (champ_confdist + champ_w80): %.4f  rho %.3f  -> rank %d of %d"
              % (v, rho, rank, len(pairs)))
    else:
        print("\n  shipped champion pair not both present in this group")
    print("\n  best solo overall %.4f | best pair overall %.4f" % (max(solo.values()), pairs[0][0]))


if __name__ == "__main__":
    main()
