"""Is a per-finding blend weight worth anything, and can 58 studies even tell?

Usage: python _blend_weight.py <gold_npz_A> <gold_npz_B>   (A is the champion, B the partner)
"""
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
RNG = np.random.default_rng(0)
NBOOT = 2000


def ranks(p):
    return np.stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])], axis=1)


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main(fa, fb):
    A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
    assert np.array_equal(A["ids"], B["ids"]), "different study order"
    ra, rb = ranks(A["pred"]), ranks(B["pred"])
    truth = (A["truth"] >= 0.5).astype(int)
    n, nf = truth.shape

    print(f"{n} studies | champion {fa.split('/')[-1]} | partner {fb.split('/')[-1]}\n")
    print(f"{'finding':4s} {'w*':>6s} {'AUC@w*':>8s} {'AUC@0.5':>8s} {'in-sample':>10s} "
          f"{'oob gain':>9s} {'win rate':>9s}")

    flat_half, tuned = [], []
    for j in range(nf):
        y = truth[:, j]
        if not (0 < y.sum() < n):
            continue
        sc = {w: auc(y, w * ra[:, j] + (1 - w) * rb[:, j]) for w in GRID}
        wbest = max(sc, key=sc.get)
        half = sc[0.5]
        flat_half.append(half)
        tuned.append(sc[wbest])

        # Refit the weight on each resample, score on the studies it did not see.
        gains, wins = [], 0
        for _ in range(NBOOT):
            idx = RNG.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), idx)
            if len(oob) < 8 or not (0 < y[idx].sum() < len(idx)) or not (0 < y[oob].sum() < len(oob)):
                continue
            insc = {w: auc(y[idx], w * ra[idx, j] + (1 - w) * rb[idx, j]) for w in GRID}
            wb = max(insc, key=lambda k: (insc[k] if insc[k] == insc[k] else -1))
            g = auc(y[oob], wb * ra[oob, j] + (1 - wb) * rb[oob, j]) - \
                auc(y[oob], 0.5 * ra[oob, j] + 0.5 * rb[oob, j])
            if g == g:
                gains.append(g)
                wins += g > 0
        g = np.mean(gains) if gains else np.nan
        print(f"{j:<4d} {wbest:6.2f} {sc[wbest]:8.4f} {half:8.4f} {sc[wbest]-half:+10.4f} "
              f"{g:+9.4f} {wins/max(1,len(gains)):9.1%}")

    print(f"\nmacro at 0.5/0.5      {np.mean(flat_half):.4f}")
    print(f"macro at tuned w      {np.mean(tuned):.4f}  (in-sample, NOT a forecast)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
