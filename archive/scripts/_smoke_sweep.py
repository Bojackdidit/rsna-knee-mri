#!/usr/bin/env python3
"""Prove the sweep path works on this box before spending an hour on it.

Loads the shipped arm, builds the eval dataset the way keval_centre_sweep does, and
runs two batches under the throttle. Checks the corpus geometry matches the arm.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu_throttle
_gpu_throttle.install()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader



def main():
    import finetune_box as F

    ARM = sys.argv[1]
    DATA, LAB = F.DATA, F.LAB
    print("[smoke] DATA =", DATA)
    print("[smoke] RSNA =", F.RSNA)

    vols = np.load(os.path.join(DATA, "all_vols.npy"), mmap_mode="r")
    print("[smoke] corpus", vols.shape, vols.dtype)

    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)

    soft = pd.read_parquet(os.environ["SWEEP_LABELS"])
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    tr = pd.read_csv(os.path.join(F.RSNA, "train.csv"))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_ids = set(tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"]) & idset
    val_ids = [u for u in ids if u in soft.index and u not in gold_ids][:8]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}
    print("[smoke] %d studies in corpus, %d gold, using %d for smoke" % (len(ids), len(gold_ids), len(val_ids)))

    ck = torch.load(ARM, map_location="cpu", weights_only=False)
    arch, res = ck.get("arch"), int(ck.get("res", 384))
    bb = F.build_backbone(arch, pretrained=False)
    model = F.RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().cuda()
    print("[smoke] arm %s | arch %s | res %d" % (os.path.basename(ARM), arch, res))

    F.CENTRE_FRAC = 1.0
    ds = F.StudyWindows(DATA, val_ids, id2row, labels, res, 12, train=False, norm="imagenet")
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=F.collate)

    t0 = time.time()
    n = 0
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = torch.sigmoid(model(x.cuda()).float())
            print("[smoke] batch %d  x%s -> out%s  peakVRAM %.2f GB"
                  % (n, tuple(x.shape), tuple(o.shape), torch.cuda.max_memory_allocated() / 1e9), flush=True)
            n += 1
            if n >= 2:
                break
    print("[smoke] OK - %d batches in %.1fs" % (n, time.time() - t0))


if __name__ == "__main__":
    main()
