#!/usr/bin/env python3
"""Build an improved training label set from the severity distributions. CPU only, no GPU."""
import argparse, json
import numpy as np, pandas as pd

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


def rankpct(x):
    o = x.argsort().argsort().astype(np.float64)
    return o / max(1, len(x) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="phase2/labels_v2/conf_all.json")
    ap.add_argument("--ship", default="phase2/box_stage/labeler/labels_confdist.parquet")
    ap.add_argument("--out", default="labels_sevblend.parquet")
    ap.add_argument("--calibrate", action="store_true",
                    help="rescale each finding to the gold prevalence and lift the confidently-"
                         "silent group to its measured positive rate (Synovitis 0.174, PF OA 0.143)")
    ap.add_argument("--silence_floor", action="store_true",
                    help="lift confidently-silent studies to their measured positive rate. MEASURED "
                         "to cost -0.0068 macro of label ranking on gold-58 - off by default.")
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--conf_gold", default="phase2/labels_v2/conf_gold_tf.json")
    ap.add_argument("--w", type=float, default=0.5, help="weight on E[s]; 1.0 = severity only")
    a = ap.parse_args()

    ca = json.load(open(a.conf))
    ship = pd.read_parquet(a.ship)
    ship["StudyInstanceUID"] = ship["StudyInstanceUID"].astype(str)
    ship = ship.set_index("StudyInstanceUID")
    ids = [u for u in ship.index if u in ca]
    print("[labels] %d studies with both severity and shipped labels" % len(ids))

    C = np.array([ca[u] for u in ids], float)                  # (N,12,4)
    if not np.allclose(C.sum(-1), 1, atol=1e-3):
        raise SystemExit("severity rows do not sum to 1 - not a distribution")
    es = (C[..., 1] + 2 * C[..., 2] + 3 * C[..., 3]) / 3.0      # E[s] scaled to [0,1]
    S = ship.loc[ids, LAB].values.astype(float)

    out = np.zeros_like(S)
    print("\n%-18s %8s %8s %8s" % ("finding", "rho", "mean_out", "note"))
    from scipy.stats import spearmanr
    for j, l in enumerate(LAB):
        r = spearmanr(S[:, j], es[:, j]).statistic
        # rank-blend, then rescale to the shipped label's own mean
        b = a.w * rankpct(es[:, j]) + (1 - a.w) * rankpct(S[:, j])
        b = b / max(b.mean(), 1e-6) * S[:, j].mean()
        out[:, j] = np.clip(b, 0.0, 1.0)
        note = ""
        if r < 0.60:
            note = "LOW rho - sources disagree"
        if l == "Synovitis":
            note = "63%% mushy in shipped; weak in both"
        print("%-18s %8.3f %8.3f  %s" % (l, r, out[:, j].mean(), note))

    if a.calibrate:
        # prevalence calibration + silence floor, both fitted on the 58 gold studies
        import json as _json
        _tr = pd.read_csv(a.train_csv); _tr["StudyInstanceUID"] = _tr["StudyInstanceUID"].astype(str)
        _g = _tr[_tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
        _cg = _json.load(open(a.conf_gold))
        _ids = [u for u in _g.index if u in _cg]
        _C = np.array([_cg[u] for u in _ids], float)
        _Y = _g.loc[_ids, LAB].values.astype(int)
        print("")
        print("[calibrate] on %d gold studies" % len(_ids))
        print("%-18s %9s %9s %9s %9s" % ("finding", "gold prev", "was", "now", "silent floor"))
        for j, l in enumerate(LAB):
            prev = float(_Y[:, j].mean())
            sil = _C[:, j, 0] > 0.9
            floor = float(_Y[sil, j].mean()) if sil.sum() >= 5 else 0.0
            before = float(out[:, j].mean())
            # rescale to gold prevalence, then lift the confidently-silent group
            out[:, j] = np.clip(out[:, j] / max(before, 1e-6) * prev, 0.0, 1.0)
            if a.silence_floor:
                silent_all = C[:, j, 0] > 0.9
                out[silent_all, j] = np.maximum(out[silent_all, j], floor)
            print("%-18s %9.3f %9.3f %9.3f %9.3f" % (l, prev, before, out[:, j].mean(), floor))

    df = pd.DataFrame(out, columns=LAB)
    df.insert(0, "StudyInstanceUID", ids)
    df.to_parquet(a.out, index=False)
    print("\n[labels] wrote %s  (%d rows x %d cols, w=%.2f on E[s])"
          % (a.out, len(df), len(df.columns), a.w))
    print("[labels] drop in with:  --labels %s" % a.out)


if __name__ == "__main__":
    main()
