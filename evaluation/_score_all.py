#!/usr/bin/env python3
"""Score every 384-res arm on both clean instruments (870 fold + gold-58) and dump predictions with provenance."""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu_throttle

OUT = os.environ.get("SCORE_OUT", "gate2026")
K = int(os.environ.get("SCORE_K", 48))
RES_WANTED = int(os.environ.get("SCORE_RES", 384))


def macro(P, Y):
    a = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j])
         for j in range(Y.shape[1]) if len(set((Y[:, j] >= 0.5).astype(int).tolist())) > 1]
    return (float(np.mean(a)), len(a)) if a else (float("nan"), 0)


def main():
    _gpu_throttle.install()
    import finetune_box as F

    os.makedirs(OUT, exist_ok=True)
    jsonl = os.path.join(OUT, "scores.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            try:
                done.add(json.loads(line)["arm"])
            except Exception:
                pass
    print("[scoreall] %d arms already done, skipping those" % len(done), flush=True)

    DATA, LAB = F.DATA, F.LAB
    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)

    tr = pd.read_csv(os.path.join(F.RSNA, "train.csv"))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    gold_lab = {u: gold_df.loc[u, LAB].values.astype(np.float32) for u in gold_ids}

    soft = pd.read_parquet(os.environ["SWEEP_LABELS"])
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    train_ids = [u for u in ids if u in soft.index and u not in set(gold_ids)]

    gdf = pd.read_csv(os.environ["SWEEP_GROUP"])
    gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
    fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
    sizes = {}
    for u in train_ids:
        f = fp_of.get(u)
        if f is not None:
            sizes[f] = sizes.get(f, 0) + 1
    load, fold_of_fp = [0] * 5, {}
    for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        j = int(np.argmin(load))
        fold_of_fp[f] = j
        load[j] += n
    held = {f for f, j in fold_of_fp.items() if j == 0}
    val_ids = [u for u in train_ids if fp_of.get(u) in held]
    soft_lab = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}
    print("[scoreall] gold %d studies | 870-fold %d studies | K=%d" % (len(gold_ids), len(val_ids), K), flush=True)

    arms = {}
    for f in sorted(glob.glob(os.path.join(".", "**", "raptor_ft_*.pt"), recursive=True)):
        arms.setdefault(os.path.basename(f)[:-3], f)

    def score(model, res, sids, labs):
        ds = F.StudyWindows(DATA, sids, id2row, labs, res, K, train=False, norm="imagenet")
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=F.collate)
        P, Y = [], []
        with torch.no_grad():
            for x, y in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = torch.sigmoid(model(x.cuda()).float())
                P.append(o.cpu().numpy())
                Y.append(y.numpy())
        return np.concatenate(P), np.concatenate(Y)

    t_start = time.time()
    for i, (tag, path) in enumerate(sorted(arms.items())):
        if tag in done:
            continue
        t0 = time.time()
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            res = int(ck.get("res", 0))
            arch = str(ck.get("arch"))
            if res != RES_WANTED:
                continue
            bb = F.build_backbone(arch, pretrained=False)
            model = F.RaptorClassifier(bb, F_dim=bb.num_features)
            model.load_state_dict(ck["model"], strict=True)
            model.eval().cuda()
            del ck

            Pg, Yg = score(model, res, gold_ids, gold_lab)
            mg, ng = macro(Pg, Yg)
            Ps, Ys = score(model, res, val_ids, soft_lab)
            ms, ns = macro(Ps, Ys)

            np.savez(os.path.join(OUT, "gold58_%s.npz" % tag), pred=Pg, truth=Yg,
                     ids=np.array(gold_ids, dtype=object))
            np.savez(os.path.join(OUT, "fold870_%s.npz" % tag), pred=Ps, truth=Ys,
                     ids=np.array(val_ids, dtype=object))
            rec = {"arm": tag, "path": path, "arch": arch, "res": res, "K": K,
                   "gold58": mg, "gold58_n": ng, "fold870": ms, "fold870_n": ns,
                   "delta": ms - mg, "labels": os.path.basename(os.environ["SWEEP_LABELS"]),
                   "grade_fold": 0, "secs": round(time.time() - t0, 1)}
            with open(jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("[%3d] %-34s %-30s gold58 %.4f | 870 %.4f | delta %+.4f | %.0fs"
                  % (i, tag, arch.split(".")[0][:30], mg, ms, ms - mg, time.time() - t0), flush=True)
            del model, bb
            torch.cuda.empty_cache()
        except Exception as e:
            print("[%3d] %-34s FAILED: %s" % (i, tag, str(e)[:90]), flush=True)
            torch.cuda.empty_cache()

    print("[scoreall] finished in %.1f min" % ((time.time() - t_start) / 60), flush=True)


if __name__ == "__main__":
    main()
