#!/usr/bin/env python3
"""PER-SLICE training. The falsification test for why we cap at 0.941.

THE THESIS. Our bag trainer gives the encoder ONE study-level gradient per study per epoch over
k sampled windows - at k=6 that is ~6 window views and 1 loss per study. The teams above 0.95 train
a loss on EVERY slice of the full stack, then aggregate. Same corpus, same labels, ~10-60x the
supervised throughput through the encoder. That predicts exactly what we measure: a backbone that
barely moves from ImageNet init (weight distance 0.0220 vs the head 0.8953) even at the
proven-optimal LR, EfficientNet-B0 working for them and collapsing to 0.896 for us, and 337 teams
piled at 0.941-0.943 permuting weights inside the same regime.

THE TEST. Identical arch, resolution, fold, labels, augmentation and split to the bag control E0
(b0@288, fold 0, reclaim 0.7, aug -> 0.8961 on the 350-study gate). The ONLY change is the
supervision regime: one BCE per SLICE instead of one per study bag. Study score at eval is an
aggregate over every valid slice.

READ: >= +0.015 over 0.8961 confirms the mechanism. Below that refutes it, cheaply.

Usage: finetune_slice.py --arch ARCH --res R --epochs N --bs B --agg max|mean|topk --tag TAG
"""
import argparse, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import finetune_box as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, LAB, RSNA = F.DATA, F.LAB, F.RSNA
FULL_SLOTS = F.FULL_SLOTS


class SliceSamples(Dataset):
    """One sample = ONE slice triplet, labelled with its study label.

    A centre is admissible only if c-1, c and c+1 are all valid AND all inside the same slot: the
    volume concatenates five separate MRI series, so a triplet spanning a boundary stacks two
    different acquisitions into one image.
    """

    def __init__(self, root, ids, id2row, labels, res, train, aug=True):
        self.root, self.ids, self.id2row, self.labels = root, ids, id2row, labels
        self.res, self.train, self.aug = res, train, aug
        self.vols = self.masks = None
        self.index = None
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _ensure(self):
        if self.vols is None:
            self.vols = np.load(os.path.join(self.root, "all_vols.npy"), mmap_mode="r")
            self.masks = np.load(os.path.join(self.root, "all_masks.npy"))
        if self.index is None:
            self.index = []
            D = self.vols.shape[1]
            keep = getattr(F, "KEEP_SLOTS", None)
            slots = [FULL_SLOTS[i] for i in keep] if keep else list(FULL_SLOTS)
            slots = [(a, b) for a, b in slots if a < D]
            for uid in self.ids:
                r = self.id2row[uid]
                vs = set(int(v) for v in np.where(self.masks[r] > 0)[0])
                for a, b in slots:
                    for c in range(a + 1, min(b, D) - 1):
                        if (c - 1) in vs and c in vs and (c + 1) in vs:
                            self.index.append((r, c, uid))

    def __len__(self):
        self._ensure()
        return len(self.index)

    def __getitem__(self, i):
        self._ensure()
        r, c, uid = self.index[i]
        vol = self.vols[r]
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != self.res:
            t = Fn.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                               align_corners=False)[0]
        if self.train and self.aug and hasattr(F, "apply_aug"):
            t = F.apply_aug(t[None])[0]
        t = (t - self._mean) / self._std
        return t, torch.from_numpy(self.labels[uid]), i


class SliceNet(nn.Module):
    """Backbone plus a plain linear head. There is nothing to pool: one slice, one prediction."""

    def __init__(self, backbone, F_dim, n=12, drop=0.1):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(F_dim, n)

    def forward(self, x):
        return self.fc(self.drop(self.norm(self.backbone(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--res", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--slots", default="")
    ap.add_argument("--aug", default="geo,erase,blur,noise")
    ap.add_argument("--labels", default="/workspace/knee/labels_opus_blend.parquet")
    ap.add_argument("--val_ids", default="/workspace/knee/gate350_E0.csv")
    ap.add_argument("--agg", default="max")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="slice")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.slots:
        F.KEEP_SLOTS = [int(x) for x in a.slots.split(",") if x.strip() != ""]
    F.AUG_SET = set(x for x in a.aug.split(",") if x)
    print("[slice] slots=%s aug=%s" % (getattr(F, "KEEP_SLOTS", None), sorted(F.AUG_SET)), flush=True)

    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    tr = pd.read_csv(os.path.join(RSNA, "train.csv"))
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = set(tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"]) & set(ids)
    soft = pd.read_parquet(a.labels)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")

    val = [x.strip() for x in open(a.val_ids).read().split()
           if x.strip() and x.strip() != "StudyInstanceUID"]
    val = [u for u in val if u in soft.index]
    vset = set(val)
    trn = [u for u in ids if u in soft.index and u not in gold and u not in vset]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in trn + val}
    print("[slice] train %d studies | val %d studies (the bag control split, exactly)"
          % (len(trn), len(val)), flush=True)

    dtr = SliceSamples(DATA, trn, id2row, labels, a.res, True)
    dva = SliceSamples(DATA, val, id2row, labels, a.res, False)
    ntr, nva = len(dtr), len(dva)
    print("[slice] train SLICES %d | val SLICES %d" % (ntr, nva), flush=True)
    print("[slice] supervision density %.1f losses per study per epoch (bag regime = 1.0)"
          % (ntr / max(len(trn), 1)), flush=True)

    ltr = DataLoader(dtr, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                     pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    lva = DataLoader(dva, batch_size=a.bs * 2, shuffle=False, num_workers=a.workers, pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = F.build_backbone(a.arch, pretrained=True)
    model = SliceNet(bb, bb.num_features).to(dev)
    print("[slice] %s @ %d | params %s"
          % (a.arch, a.res, "{:,}".format(sum(p.numel() for p in model.parameters()))), flush=True)

    prev = np.clip(np.stack([labels[u] for u in trn]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * max(len(ltr), 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")

    dva._ensure()
    vuid = np.array([dva.index[i][2] for i in range(nva)])
    Yv = np.stack([labels[u] for u in val])
    masks = {u: (vuid == u) for u in val}

    best = 0.0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = n = 0
        for x, y, _ in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = lossf(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
            n += 1
        model.eval()
        P = []
        with torch.no_grad():
            for x, _, _ in lva:
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    P.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
        P = np.concatenate(P) if P else np.zeros((0, 12), np.float32)
        res = {}
        for agg in ("max", "mean", "topk"):
            S = np.zeros((len(val), 12), np.float32)
            for i, u in enumerate(val):
                m = masks[u]
                if not m.any():
                    continue
                p = P[m]
                if agg == "max":
                    S[i] = p.max(0)
                elif agg == "mean":
                    S[i] = p.mean(0)
                else:
                    k = max(1, int(round(0.2 * len(p))))
                    S[i] = np.sort(p, 0)[-k:].mean(0)
            au = [roc_auc_score((Yv[:, j] >= 0.5).astype(int), S[:, j]) for j in range(12)
                  if len(set(((Yv[:, j] >= 0.5).astype(int)).tolist())) > 1]
            res[agg] = float(np.mean(au))
        cur = res[a.agg]
        star = ""
        if cur > best:
            best = cur
            star = " *"
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "arch": a.arch, "res": a.res, "regime": "per-slice", "agg": a.agg,
                        "slots": list(F.KEEP_SLOTS) if getattr(F, "KEEP_SLOTS", None) else None,
                        "epoch": ep, "gold_auc": cur},
                       os.path.join(HERE, "raptor_slice_%s.pt" % a.tag))
        print("ep%-2d loss %.4f | max %.4f mean %.4f topk %.4f | best %.4f%s | %.0fs"
              % (ep, tot / max(n, 1), res["max"], res["mean"], res["topk"], best, star,
                 time.time() - t0), flush=True)

    print("", flush=True)
    print("DONE %s | BEST %.4f (%s)" % (a.tag, best, a.agg), flush=True)
    print("BAG CONTROL, same arch/res/fold/labels/split: 0.8961", flush=True)
    print("DELTA %+.4f -> %s" % (best - 0.8961,
                                 "MECHANISM CONFIRMED" if best - 0.8961 >= 0.015
                                 else "below the +0.015 bar, thesis not supported"), flush=True)


if __name__ == "__main__":
    main()
