"""Per-finding blend weight, evaluated OUT-OF-BAG, for the SHIPPED pair.

Shipped kernel: SCRATCH_AUG2 + CONVNEXT_E24, rank-mean at w=0.55, one weight for all 12.
Question: does any single finding want a different weight?

12 weights fit on 58 studies is the overfit that retired the gold-58 gate, so no number
here is read off the in-sample optimum. Each bootstrap fits w on in-bag studies and scores
it on the studies it did not see. Win rate = fraction of resamples beating 0.55 out-of-bag.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Synovitis','Medial Cartilage','Lateral Cartilage','PF Cartilage','Effusion']
W_SHIP = 0.55
GRID = np.arange(0.0, 1.01, 0.05)

def load(t):
    d = np.load(f"gold_oof/raptor_gold_{t}.npz", allow_pickle=True)
    return d["ids"], d["pred"], d["truth"]

ia, pa, ta = load("SCRATCH_AUG2")
ib, pb, tb = load("CONVNEXT_E24")
common = [u for u in ia if u in set(ib)]
xa = {u: i for i, u in enumerate(ia)}; xb = {u: i for i, u in enumerate(ib)}
A = np.array([pa[xa[u]] for u in common]); B = np.array([pb[xb[u]] for u in common])
Y = np.array([ta[xa[u]] for u in common])
N, F = A.shape
print(f"{N} studies, {F} findings\n")

def blend_auc(idx, f, w):
    """Rank-blend on this subset, AUC for one finding. Ranks recomputed within the subset."""
    y = Y[idx, f]
    if len(set(y.round().astype(int))) < 2: return None
    ra = rankdata(A[idx, f]) / len(idx); rb = rankdata(B[idx, f]) / len(idx)
    return roc_auc_score(y.round().astype(int), w * ra + (1 - w) * rb)

rng = np.random.default_rng(0)
NB = 400
print(f"{'finding':<20}{'w*(full)':>9}{'oob gain':>10}{'win rate':>10}{'ship AUC':>10}")
res = {}
for f in range(F):
    full = [(w, blend_auc(np.arange(N), f, w)) for w in GRID]
    full = [(w, a) for w, a in full if a is not None]
    if not full: print(f"{LAB[f]:<20}  (single-class, skipped)"); continue
    wstar = max(full, key=lambda t: t[1])[0]
    ship = dict(full).get(W_SHIP) or blend_auc(np.arange(N), f, W_SHIP)

    gains, wins = [], 0
    for _ in range(NB):
        ib_ = rng.integers(0, N, N)
        oob = np.array(sorted(set(range(N)) - set(ib_.tolist())))
        if len(oob) < 12: continue
        cand = [(w, blend_auc(ib_, f, w)) for w in GRID]
        cand = [(w, a) for w, a in cand if a is not None]
        if not cand: continue
        w_fit = max(cand, key=lambda t: t[1])[0]
        g_fit, g_ship = blend_auc(oob, f, w_fit), blend_auc(oob, f, W_SHIP)
        if g_fit is None or g_ship is None: continue
        gains.append(g_fit - g_ship); wins += (g_fit > g_ship)
    if not gains: continue
    g = float(np.mean(gains)); wr = wins / len(gains)
    res[LAB[f]] = (wstar, g, wr)
    print(f"{LAB[f]:<20}{wstar:>9.2f}{g:>+10.4f}{wr:>9.1%}{ship:>10.4f}")

print("\nA per-finding weight is only worth shipping where the out-of-bag gain is positive")
print("AND the win rate is high; a positive gain at ~50% win rate is noise.")
