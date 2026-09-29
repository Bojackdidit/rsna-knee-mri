#!/usr/bin/env python3
"""Weight-space soup of two independently trained CoAtNet arms, scored on gold-58.

Both SCRATCH_AUG2 and FULL96 are coatnet_rmlp_2_rw_384 fine-tuned from the SAME timm checkpoint
(src=timm-pretrained), so they may lie in one loss basin and a straight parameter average may hold
up. Worth testing because a soup is free at inference - one model instead of two - which would free
a whole runtime slot for a third arm inside the 1 h cap.

This is NOT the SWA that is already refuted here: that averaged epochs within one run. This averages
two separate runs.

Runs at low K and a small VRAM cap so it can share the card with fold training. K is held constant
across all three models, so the comparison is like-for-like even though the absolute numbers are
below the K=48 figures.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import _gpu_throttle
    _gpu_throttle.install_vram_only()
    import finetune_box as F

    K = int(os.environ.get("SOUP_K", 24))
    A = os.environ["SOUP_A"]
    B = os.environ["SOUP_B"]

    DATA, LAB = F.DATA, F.LAB
    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    tr = pd.read_csv(os.path.join(F.RSNA, "train.csv"))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
    gids = [u for u in gold.index if u in set(ids)]
    labels = {u: gold.loc[u, LAB].values.astype(np.float32) for u in gids}
    print("[soup] %d gold studies | K=%d" % (len(gids), K), flush=True)

    ckA = torch.load(A, map_location="cpu", weights_only=False)
    ckB = torch.load(B, map_location="cpu", weights_only=False)
    assert ckA["arch"] == ckB["arch"] and ckA["res"] == ckB["res"], "arch/res mismatch"
    sdA, sdB = ckA["model"], ckB["model"]
    assert set(sdA) == set(sdB), "state dict keys differ"

    # distance in weight space: if these are in different basins a soup will collapse
    num = sum(float(((sdA[k].float() - sdB[k].float()) ** 2).sum()) for k in sdA if sdA[k].is_floating_point())
    den = sum(float((sdA[k].float() ** 2).sum()) for k in sdA if sdA[k].is_floating_point())
    print("[soup] relative weight distance %.4f" % (num ** 0.5 / den ** 0.5), flush=True)

    def score(sd, tag):
        bb = F.build_backbone(ckA["arch"], pretrained=False)
        m = F.RaptorClassifier(bb, F_dim=bb.num_features)
        m.load_state_dict(sd, strict=True)
        m.eval().cuda()
        ds = F.StudyWindows(DATA, gids, id2row, labels, int(ckA["res"]), K, train=False, norm="imagenet")
        dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=F.collate)
        P, Y = [], []
        with torch.no_grad():
            for x, y in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    P.append(torch.sigmoid(m(x.cuda()).float()).cpu().numpy())
                Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        a = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j]) for j in range(Y.shape[1])
             if len(set((Y[:, j] >= 0.5).astype(int).tolist())) > 1]
        v = float(np.mean(a))
        print("[soup] %-22s gold-58 %.4f  (peak %.2f GB)" % (tag, v, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        del m, bb
        torch.cuda.empty_cache()
        return v

    vA = score(sdA, os.path.basename(A))
    vB = score(sdB, os.path.basename(B))
    best, bw = None, None
    for w in (0.5, 0.65, 0.8):
        soup = {k: (sdA[k].float() * w + sdB[k].float() * (1 - w)).to(sdA[k].dtype)
                if sdA[k].is_floating_point() else sdA[k] for k in sdA}
        v = score(soup, "soup w=%.2f" % w)
        if best is None or v > best:
            best, bw = v, w
            torch.save({"model": soup, "arch": ckA["arch"], "res": ckA["res"],
                        "lab": ckA["lab"], "gold_auc": v, "src": "soup"}, "raptor_ft_SOUP.pt")
    print("\n[soup] best soup %.4f at w=%.2f | A %.4f | B %.4f" % (best, bw, vA, vB), flush=True)
    print("[soup] soup beats the stronger arm: %s" % (best > max(vA, vB)), flush=True)


if __name__ == "__main__":
    main()
