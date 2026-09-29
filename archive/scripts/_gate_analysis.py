#!/usr/bin/env python3
"""Turn the overnight dumps into a decision. CPU only.

Three questions, in order:

  1. WHICH ARMS ARE HONEST. delta = macro(870) - macro(gold58). Gold-58 is held out of training
     for every arm; the 870 fold is not. A large positive delta is memorisation, not quality.
  2. WHICH PAIR IS BEST, scored only on gold-58, the instrument valid across the whole pool.
  3. DOES THE BEST PAIR BEAT THE SHIPPED CHAMPION BY MORE THAN NOISE. Gold-58 is 58 studies, so a
     point estimate means nothing on its own. Paired bootstrap over studies gives a win rate; the
     lane's own standard is that a lever must hold in the large majority of resamples, not merely
     win on the mean.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

OUT = os.environ.get("SCORE_OUT", "gate2026")
BOOT = int(os.environ.get("BOOT", 2000))
rng = np.random.default_rng(12345)


def macro_idx(P, Y, idx):
    a = []
    for j in range(Y.shape[1]):
        y = (Y[idx, j] >= 0.5).astype(int)
        if len(set(y.tolist())) > 1:
            a.append(roc_auc_score(y, P[idx, j]))
    return float(np.mean(a)) if a else np.nan


def main():
    recs = {}
    jl = os.path.join(OUT, "scores.jsonl")
    for line in open(jl, encoding="utf-8"):
        try:
            r = json.loads(line)
            recs[r["arm"]] = r
        except Exception:
            pass
    print("arms scored: %d\n" % len(recs))

    ds = sorted((r["delta"], r["arm"], r["gold58"], r["fold870"]) for r in recs.values())
    med = float(np.median([d for d, _, _, _ in ds]))
    print("=== contamination detector: delta = 870 - gold58   (median %+.4f) ===" % med)
    print("  %-36s %8s %8s %8s" % ("arm", "gold58", "870", "delta"))
    for d, a, g, s in ds[-10:][::-1]:
        flag = "  <-- MEMORISED" if d > med + 0.02 else ""
        print("  %-36s %8.4f %8.4f %+8.4f%s" % (a, g, s, d, flag))
    print("  ... lowest deltas ...")
    for d, a, g, s in ds[:5]:
        print("  %-36s %8.4f %8.4f %+8.4f" % (a, g, s, d))

    G = {}
    for f in glob.glob(os.path.join(OUT, "gold58_*.npz")):
        d = np.load(f, allow_pickle=True)
        tag = os.path.basename(f)[7:-4]
        ids = np.asarray(d["ids"], dtype=object).astype(str)
        o = np.argsort(ids)
        G[tag] = (d["pred"][o], d["truth"][o])
    if not G:
        print("\nno gold58 dumps yet")
        return
    tags = sorted(G)
    Y = G[tags[0]][1]
    n = Y.shape[0]
    R = {t: np.column_stack([rankdata(G[t][0][:, j]) for j in range(G[t][0].shape[1])]) for t in tags}

    full = np.arange(n)
    solo = {t: macro_idx(G[t][0], Y, full) for t in tags}
    print("\n=== solo on gold-58 (clean for every arm), best 12 ===")
    for t, v in sorted(solo.items(), key=lambda kv: -kv[1])[:12]:
        print("  %-36s %.4f  (delta %+.4f)" % (t, v, recs.get(t, {}).get("delta", float("nan"))))

    pairs = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            pairs.append((macro_idx(0.5 * R[a] + 0.5 * R[b], Y, full), a, b))
    pairs.sort(reverse=True)
    print("\n=== best 12 PAIRS on gold-58 (rank-mean) ===")
    for v, a, b in pairs[:12]:
        print("  %-56s %.4f" % (a + " + " + b, v))

    champ = [p for p in pairs if "confdist" in (p[1] + p[2]) and "w80" in (p[1] + p[2])]
    if not champ:
        print("\nchampion pair not found among dumps - cannot run the head-to-head")
        return
    cv, ca, cb = champ[0]
    print("\nshipped champion pair: %s + %s = %.4f (rank %d of %d)"
          % (ca, cb, cv, pairs.index(champ[0]) + 1, len(pairs)))

    print("\n=== paired bootstrap vs the champion pair, %d resamples of the 58 studies ===" % BOOT)
    Cp = 0.5 * R[ca] + 0.5 * R[cb]
    print("  %-56s %8s %8s" % ("challenger", "mean", "win%"))
    for v, a, b in pairs[:8]:
        if (a, b) == (ca, cb):
            continue
        Xp = 0.5 * R[a] + 0.5 * R[b]
        wins, diffs = 0, []
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            if len(set(Y[idx, 0].tolist())) < 2:
                continue
            dv = macro_idx(Xp, Y, idx) - macro_idx(Cp, Y, idx)
            if not np.isnan(dv):
                diffs.append(dv)
                wins += dv > 0
        if diffs:
            print("  %-56s %+8.4f %7.1f%%" % (a + " + " + b, float(np.mean(diffs)), 100.0 * wins / len(diffs)))


if __name__ == "__main__":
    main()
