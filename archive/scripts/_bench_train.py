#!/usr/bin/env python3
"""What would finishing the MaxViT arm actually cost on this box? Measure a real training step.

The arm is the only meaningfully decorrelated family we have (rho 0.78 against everything, vs
0.888 for the shipped pair) and it stopped at epoch 4 while the pool median is 13. Whether that
is worth resuming is a question about hours, so measure hours instead of estimating them:
forward + backward + optimiser at the real training geometry, then extrapolate over the corpus.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu_throttle


def main():
    _gpu_throttle.install_vram_only()
    import finetune_box as F

    ARM = sys.argv[1]
    K = int(os.environ.get("TRAIN_K", 12))
    BS = int(os.environ.get("TRAIN_BS", 2))
    STEPS = int(os.environ.get("TRAIN_STEPS", 12))
    GRAD_CKPT = os.environ.get("GRAD_CKPT", "1") == "1"

    DATA, LAB = F.DATA, F.LAB
    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    soft = pd.read_parquet(os.environ["SWEEP_LABELS"])
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    sids = [u for u in ids if u in soft.index][:200]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in sids}
    n_corpus = sum(1 for u in ids if u in soft.index)

    ck = torch.load(ARM, map_location="cpu", weights_only=False)
    arch, res = str(ck["arch"]), int(ck["res"])
    bb = F.build_backbone(arch, pretrained=False)
    model = F.RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.train().cuda()
    if GRAD_CKPT:
        try:
            bb.set_grad_checkpointing(True)
            print("[bench] grad checkpointing ON", flush=True)
        except Exception as e:
            print("[bench] grad checkpointing unavailable: %s" % str(e)[:60], flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    lossf = torch.nn.BCEWithLogitsLoss()

    ds = F.StudyWindows(DATA, sids, id2row, labels, res, K, train=True, norm="imagenet")
    dl = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=F.collate)

    print("[bench] %s | res %d | K_train %d | bs %d | corpus %d studies"
          % (arch.split(".")[0], res, K, BS, n_corpus), flush=True)

    it = iter(dl)
    x, y = next(it)                                   # warm up, untimed
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lossf(model(x.cuda()), y.cuda())
    loss.backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0, n = time.perf_counter(), 0
    for _ in range(STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(x.cuda()), y.cuda())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        n += x.shape[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    sps = n / dt
    epoch_h = n_corpus / sps / 3600.0
    print("[bench] %.2f studies/s | peak VRAM %.2f GB" % (sps, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print("[bench] ONE EPOCH over %d studies = %.2f h unthrottled, %.2f h at duty 0.90"
          % (n_corpus, epoch_h, epoch_h / 0.90), flush=True)
    for e in (8, 12):
        print("[bench]   +%2d epochs (to epoch %2d) = %5.1f h at duty 0.90" % (e, 4 + e, e * epoch_h / 0.90), flush=True)


if __name__ == "__main__":
    main()
