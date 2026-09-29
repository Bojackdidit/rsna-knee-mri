#!/usr/bin/env python3
"""How fast can this box score one candidate arm? Measure, do not estimate.

One "test" = scoring an arm on the 870-study grouped holdout. Everything downstream
(arm ranking, pair selection, geometry sweeps) is built out of that unit, so its cost
sets how many questions a day we can actually answer.

Times a fixed number of batches per setting and extrapolates to the full 870.
Run through _run_sweep.py so the __main__ guard makes DataLoader workers safe here.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import finetune_box as F

import _gpu_throttle


def main():
    _gpu_throttle.install_vram_only()
    ARM = sys.argv[1]
    N_BATCH = int(os.environ.get("BENCH_BATCHES", 24))
    K = int(os.environ.get("BENCH_K", 48))

    DATA, LAB = F.DATA, F.LAB
    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    soft = pd.read_parquet(os.environ["SWEEP_LABELS"])
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    val_ids = [u for u in ids if u in soft.index][:400]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in val_ids}

    ck = torch.load(ARM, map_location="cpu", weights_only=False)
    bb = F.build_backbone(ck["arch"], pretrained=False)
    model = F.RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().cuda()
    res = int(ck.get("res", 384))

    def timed(bs, workers, chan_last, label):
        m = model.to(memory_format=torch.channels_last) if chan_last else model
        ds = F.StudyWindows(DATA, val_ids, id2row, labels, res, K, train=False, norm="imagenet")
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=F.collate)
        it = iter(dl)
        with torch.no_grad():
            x, _ = next(it)                                    # warm up, not timed
            xx = x.cuda(non_blocking=True)
            if chan_last:
                xx = xx.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(xx)
            torch.cuda.synchronize()
            t0, n = time.perf_counter(), 0
            for _ in range(N_BATCH):
                try:
                    x, _ = next(it)
                except StopIteration:
                    break
                xx = x.cuda(non_blocking=True)
                if chan_last:
                    xx = xx.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(xx)
                n += x.shape[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        sps = n / dt
        full = 870 / sps / 60.0
        print("  %-34s %6.2f studies/s | 870-study eval %5.1f min | %4.0f evals/day"
              % (label, sps, full, 1440 / full), flush=True)
        del dl
        return sps


    print("[bench] K=%d res=%d | %d batches per setting | VRAM cap in force" % (K, res, N_BATCH), flush=True)
    grid = [(1,0,False,"K48 bs1  workers0"),
            (1,4,False,"K48 bs1  workers4"),
            (2,4,False,"K48 bs2  workers4")]
    for bs, w, cl, lab in grid:
        try:
            timed(bs, w, cl, lab)
        except Exception as e:
            print("  %-34s FAILED: %s" % (lab, str(e)[:70]), flush=True)
            torch.cuda.empty_cache()
    print("[bench] peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)


if __name__ == "__main__":
    main()
