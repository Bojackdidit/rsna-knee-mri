#!/usr/bin/env python3
"""Fine-tune Raptor (our SSL knee encoder) on the 12-finding RSNA knee task."""
import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

HERE = os.path.dirname(os.path.abspath(__file__))          # my workspace: ckpt + outputs
# DATA/labels live in MAIN's scratch (read-only reads); outputs stay in HERE (my workspace).
DATA = os.environ.get("KNEE_DATA_ROOT",
    r"C:/Users/johna/AppData/Local/Temp/claude/D--Artificial-Nervous-System/efa12c9d-7852-4b73-9032-6dc08d47204d/scratchpad/rsna_knee/raptor")
RSNA = os.path.dirname(DATA)  # .../rsna_knee (train.csv, labeler/)
CENTRE_FRAC = 1.0   # set from --centre_frac; 1.0 = original full-range sampling
CROP_FRAC = 1.0     # set from --crop_frac; centre-crop of H,W before resize (joint magnification)
CORPUS_SLOTS = [(0, 12), (12, 22), (22, 30), (30, 36), (36, 44)]
KEEP_SLOTS = None   # set from --slots, e.g. '0,2,4' keeps the three fat-suppressed series
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


# ------------------------------- data ----------------------------------------
class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none"):
        self.root = root
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols = np.load(os.path.join(self.root, "all_vols.npy"), mmap_mode="r")   # (N,D,H,W) u8
            self.masks = np.load(os.path.join(self.root, "all_masks.npy"))                 # (N,D) u8

    def _centers(self, valid):
        # valid slice indices; window centers must have both neighbors valid & in-range
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        # CENTRE_FRAC keeps only the middle fraction of candidate centres (train and eval alike).
        if CENTRE_FRAC < 1.0 and len(cs) > 3:
            keep = max(3, int(round(len(cs) * CENTRE_FRAC)))
            mid = (len(cs) - keep) // 2
            cs = cs[mid:mid + keep]
        if self.train:
            reps = self.k // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:self.k]
        # eval: evenly spaced deterministic
        idx = np.linspace(0, len(cs) - 1, self.k).round().astype(int)
        return [cs[i] for i in idx]

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        if KEEP_SLOTS is not None:
            _keep = np.zeros(len(m), bool)
            for _si in KEEP_SLOTS:
                _a, _b = CORPUS_SLOTS[_si]
                _keep[_a:min(_b, len(m))] = True
            m = (np.asarray(m) > 0) & _keep
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        cs = self._centers(valid)
        wins = np.empty((len(cs), 3, self.res, self.res), np.float32)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        for j, c in enumerate(cs):
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            t = torch.from_numpy(tri)
            # ROI crop: keep the middle CROP_FRAC of H,W. Train jitters the centre +/-4%; eval takes the exact centre.
            if CROP_FRAC < 1.0:
                _H, _W = int(t.shape[-2]), int(t.shape[-1])
                _ch, _cw = int(round(_H * CROP_FRAC)), int(round(_W * CROP_FRAC))
                if self.train and self.aug:
                    _dy = random.randint(-int(round(0.04 * _H)), int(round(0.04 * _H)))
                    _dx = random.randint(-int(round(0.04 * _W)), int(round(0.04 * _W)))
                else:
                    _dy = _dx = 0
                _y0 = max(0, min(_H - _ch, (_H - _ch) // 2 + _dy))
                _x0 = max(0, min(_W - _cw, (_W - _cw) // 2 + _dx))
                t = t[..., _y0:_y0 + _ch, _x0:_x0 + _cw]
            if t.shape[-1] != self.res:
                t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                                  align_corners=False)[0]
            wins[j] = t.numpy()
        x = torch.from_numpy(wins)  # (K,3,res,res) in [0,1]
        if self.train and self.aug:
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":       # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        return x, y


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    return xs, ys


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    is_vit = any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:                                    # ViT-family: CLS token + interpolatable pos-embed
        kw.update(global_pool="token", dynamic_img_size=True)
    else:                                         # CNN (ConvNeXt/EfficientNet/...): global avg pool, any input size
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


def load_raptor(bb, ckpt_path):
    if ckpt_path in ("timm", "pretrained"):
        # handled at build time (pretrained=True); this path only tags the source
        return "timm-pretrained"
    if ckpt_path in ("", "none", "None"):
        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)
        return "random-init"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Prefer the EMA teacher (smoother, better downstream) over the student.
    st = ck.get("teacher", ck.get("student", ck))
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"


class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))       # per-window, per-finding attention score
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):                                  # x: (B,K,3,H,W) -> (B,K,F)
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))                # (B*K, F)
        return f.view(B, K, -1)

    def head(self, feats):                                # feats: (B,K,F) -> (B,n)
        h = self.norm(feats)
        a = self.att(h)                                   # (B,K,n)
        a = torch.softmax(a, dim=1)                       # attention over windows, per finding
        pooled = torch.einsum("bkn,bkf->bnf", a, h)       # (B,n,F)
        logits = (pooled * self.clsW).sum(-1) + self.clsb # (B,n)
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


# ------------------------------- train ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "raptor_cnn_ckpt", "raptor_ssl_last.pt"))
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)          # windows/study (train)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bb_lr", type=float, default=3e-5)  # encoder LR (our SSL base < DINOv2-huge, so > 8e-6)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)       # subset train studies for a fast run
    ap.add_argument("--freeze_blocks", type=int, default=0)  # freeze first N transformer blocks
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])  # imagenet for DINOv2/ImageNet backbones
    ap.add_argument("--grad_ckpt", action="store_true")  # gradient checkpointing (memory-safe for CNNs at high res)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slots", default="",
                    help="comma-separated corpus slot indices to keep, e.g. 0,2,4 (default: all)")
    ap.add_argument("--labels", default="labels_llm_soft.parquet",
                    help="soft-label parquet: filename under labeler/, or an absolute path")
    ap.add_argument("--crop_frac", type=float, default=1.0,
                    help="centre-crop this fraction of H,W before resize (joint magnification)")
    ap.add_argument("--centre_frac", type=float, default=1.0,
                    help="keep only the middle fraction of candidate window centres")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag}", flush=True)

    # ---- ids / labels ----
    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    tr = pd.read_csv(os.path.join(RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    _lab_path = a.labels if os.path.isabs(a.labels) else os.path.join(RSNA, "labeler", a.labels)
    print(f"[labels] {_lab_path}", flush=True)
    soft = pd.read_parquet(_lab_path)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}", flush=True)

    # pos_weight from soft-label prevalence (imbalance handling)
    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    tds = StudyWindows(DATA, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm)
    vds = StudyWindows(DATA, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)

    global CENTRE_FRAC, CROP_FRAC
    CENTRE_FRAC = float(a.centre_frac)
    CROP_FRAC = float(a.crop_frac)
    global KEEP_SLOTS
    KEEP_SLOTS = [int(x) for x in a.slots.split(',') if x.strip() != ''] or None
    print(f'[slots] keeping corpus slots {KEEP_SLOTS if KEEP_SLOTS else "ALL"}', flush=True)
    print(f'[crop] centre crop {CROP_FRAC:.2f} of H,W before resize', flush=True)
    print(f"[centre] window centres restricted to the middle {CENTRE_FRAC:.2f} of the stack", flush=True)
    use_timm = a.ckpt in ("timm", "pretrained")
    bb = build_backbone(a.arch, pretrained=use_timm)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)
    if a.freeze_blocks > 0:
        for nm, p in model.backbone.named_parameters():
            for b in range(a.freeze_blocks):
                if nm.startswith(f"blocks.{b}."): p.requires_grad = False

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(len(tl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    @torch.no_grad()
    def evaluate():
        model.eval(); P = []; Y = []
        for x, y in vl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x)
                loss = lossf(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); sched.step(); tot += loss.item()
        au, aucs, P, Y = evaluate()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if au > best:
            best = au; best_P = P
            best_state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                          "arch": a.arch, "lab": LAB, "epoch": ep}
        print(f"ep{ep} loss {tot/len(tl):.3f} | GOLD macro-AUC {au:.4f} (best {best:.4f}) | {time.time()-t0:.0f}s",
              flush=True)
    _, aucs, _, Y = evaluate()
    print(f"\nDONE {src} | BEST GOLD macro-AUC {best:.4f}", flush=True)
    print("per-finding AUC (best epoch):", flush=True)
    for k, v in (best_state["aucs"] if best_state else aucs).items():
        print(f"   {k:18s} {v:.3f}", flush=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=best_P, truth=Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids)},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
