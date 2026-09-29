#!/usr/bin/env python3
"""Build an improved training label set from the severity distributions. CPU only, no GPU.

WHY
---
The labeler emits a full 4-class SEVERITY distribution per finding (`conf_all.json`, 4407 studies,
12 findings x P(s=0..3)). The shipped label set collapses that to a single confidence. Measured on
the 58 gold studies, how you collapse it is worth more than any architecture change we have ever
made:

    P(s>=1)  "presence"          macro 0.8503
    P(s>=2)  "threshold"         macro 0.8780
    P(s>=3)  "severe"            macro 0.7367
    E[s]     expected severity   macro 0.8957   <- best, and it needs NO per-finding fitting

A per-finding bootstrap-gated mix of the four scored 0.8926 - WORSE than plain E[s] - so the simple
global rule wins and there is no overfitting risk. This matters because the competition's ground
truth is defined by explicit severity thresholds (">50% of fibres", ">50%-thickness cartilage loss"),
so a presence label is answering a different question than the one being scored.

The shipped `labels_confdist` correlates only **0.754** with E[s] (Fracture just 0.516), so the two
are partly independent signals of comparable strength. This blends them per finding in RANK space,
which is the same operation the kernel already uses to combine arms.

Synovitis is handled separately: its shipped labels are 63.2% "mushy" (in [0.3,0.7]) with std 0.140
- the labeler parks nearly everyone at ~0.40 and the model gets almost no gradient. Severity is not
much better there (E[s] 0.781). Flagged, not silently averaged.

Outputs a drop-in parquet with the same 13 columns as every other label file.

Usage:
    python build_sev_labels.py --out labels_sevblend.parquet --w 0.5
"""
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
        # Rank-blend, then rescale to the shipped label's own mean so pos_weight and the loss
        # scale are unchanged. BCE reads a probability, so the range must stay comparable.
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
        # ---- prevalence calibration + silence floor, both fitted on the 58 gold studies ----
        # AUC is invariant to a monotone per-finding transform, so this CANNOT change the label's
        # own ranking quality. What it changes is the TRAINING SIGNAL: BCE drives the model toward
        # the target value, and the shipped targets are systematically under-called (ACL truth
        # 0.414 vs target 0.172; Fracture 0.310 vs 0.161). A target of 0.16 for a finding that is
        # really present 31% of the time under-trains it. Only a training run can validate this.
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
            # rescale to gold prevalence (monotone -> ranking untouched), then lift the
            # confidently-silent group to its measured positive rate instead of ~0
            out[:, j] = np.clip(out[:, j] / max(before, 1e-6) * prev, 0.0, 1.0)
            if a.silence_floor:
                # ⚠️ MEASURED COST: a hard floor TIES the confidently-silent group, and on gold-58
                # that costs -0.0068 macro of the label's own ranking (Fracture -0.036, Synovitis
                # -0.028, Medial OA -0.026). The prevalence rescale above is monotone and therefore
                # free; this is not. Off by default. Only turn it on if a board read justifies it.
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
