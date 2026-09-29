"""Raptor SSL pretraining -- DINO self-distillation on the knee-MRI corpus.

Usage:
  python pretrain_raptor.py --smoke                # tiny sanity run on CPU/GPU
  python pretrain_raptor.py --epochs 120 --bs 64   # full pretrain (background)
Outputs: ckpt/raptor_ssl_last.pt (+ _ep{N}.pt), resumable.
"""
import os, sys, math, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
import timm

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- data ------------------------------------------
class GaussianBlur:
    def __init__(self, p=0.5, r=(0.1, 2.0)): self.p, self.r = p, r
    def __call__(self, img):
        if np.random.rand() > self.p: return img
        return img.filter(ImageFilter.GaussianBlur(np.random.uniform(*self.r)))

class Solarize:
    def __init__(self, p=0.2): self.p = p
    def __call__(self, img): return ImageOps.solarize(img) if np.random.rand() < self.p else img


class DinoAug:
    """Medical-safe multi-crop DINO augmentation (no hue/saturation -- channels are slices)."""
    def __init__(self, global_sz=224, local_sz=96, n_local=6,
                 gs_scale=(0.4, 1.0), ls_scale=(0.05, 0.4)):
        self.n_local = n_local
        jit = T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8)
        flip = T.RandomHorizontalFlip(0.5)   # fine for SSL (laterality only matters at fine-tune)
        norm = T.Compose([T.ToTensor()])     # keep [0,1]; encoder normalizes internally
        self.g1 = T.Compose([T.RandomResizedCrop(global_sz, scale=gs_scale), flip, jit,
                             GaussianBlur(1.0), norm])
        self.g2 = T.Compose([T.RandomResizedCrop(global_sz, scale=gs_scale), flip, jit,
                             GaussianBlur(0.1), Solarize(0.2), norm])
        self.lc = T.Compose([T.RandomResizedCrop(local_sz, scale=ls_scale), flip, jit,
                             GaussianBlur(0.5), norm])

    def __call__(self, img):
        crops = [self.g1(img), self.g2(img)]
        crops += [self.lc(img) for _ in range(self.n_local)]
        return crops


class KneeWindows(Dataset):
    """2.5D windows from all_vols.npy (memmap). Each item = a PIL RGB image of 3 consecutive
    physically-ordered slices -> DINO multi-crop views."""
    def __init__(self, root, aug, limit=None):
        self.vols_path = os.path.join(root, "all_vols.npy")
        self.vols = None      # opened lazily PER WORKER (memmap; never pickled across spawn)
        self._D = None
        self.idx = np.load(os.path.join(root, "window_index.npy"))               # (M,2) study,center
        if limit: self.idx = self.idx[:limit]
        self.aug = aug

    def __len__(self): return len(self.idx)

    def _ensure(self):
        if self.vols is None:
            self.vols = np.load(self.vols_path, mmap_mode="r")                    # (N,D,H,W) u8
            self._D = self.vols.shape[1]

    def __getitem__(self, i):
        self._ensure()
        s, c = int(self.idx[i][0]), int(self.idx[i][1])
        c = max(1, min(c, self._D - 2))
        win = np.stack([self.vols[s, c - 1], self.vols[s, c], self.vols[s, c + 1]], -1)  # H,W,3 u8
        return self.aug(Image.fromarray(win, "RGB"))


def collate(batch):
    # batch: list of [crop0..cropK]; regroup into per-crop-index tensors, then bucket by shape.
    n = len(batch[0])
    per = [torch.stack([b[k] for b in batch]) for k in range(n)]  # each (B,3,sz,sz)
    return per


# ----------------------------- model -----------------------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=8192, hidden=2048, bottleneck=256, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, bottleneck)]
        self.mlp = nn.Sequential(*layers)
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1); self.last.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last(x)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head): super().__init__(); self.backbone, self.head = backbone, head
    def forward(self, crops):
        # crops: list of tensors, bucket contiguous same-resolution runs into single forwards.
        sizes = [c.shape[-1] for c in crops]
        out, i = [], 0
        while i < len(crops):
            j = i
            while j < len(crops) and sizes[j] == sizes[i]:
                j += 1
            x = torch.cat(crops[i:j], 0)
            out.append(self.backbone(x))
            i = j
        feats = torch.cat(out, 0)
        return self.head(feats)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_crops, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_epochs=10, n_epochs=120, student_temp=0.1, center_m=0.9):
        super().__init__()
        self.student_temp, self.center_m, self.n_crops = student_temp, center_m, n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_sched = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(max(0, n_epochs - warmup_epochs)) * teacher_temp])

    def forward(self, student_out, teacher_out, epoch):
        st = (student_out / self.student_temp).chunk(self.n_crops)
        tt = self.teacher_temp_sched[min(epoch, len(self.teacher_temp_sched) - 1)]
        te = F.softmax((teacher_out - self.center) / tt, dim=-1).detach().chunk(2)  # 2 globals
        loss, n = 0.0, 0
        for iq, q in enumerate(te):                     # teacher global views
            for iv, v in enumerate(st):                 # student all views
                if iv == iq:                            # skip same view
                    continue
                loss += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                n += 1
        self._update_center(teacher_out)
        return loss / max(1, n)

    @torch.no_grad()
    def _update_center(self, teacher_out):
        c = teacher_out.mean(0, keepdim=True)
        self.center.mul_(self.center_m).add_(c * (1 - self.center_m))


def cosine_sched(base, final, epochs, steps_per_epoch, warmup_epochs=0, start=0):
    w = warmup_epochs * steps_per_epoch
    warm = np.linspace(start, base, w) if w else np.array([])
    iters = np.arange(epochs * steps_per_epoch - w)
    cos = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate([warm, cos])


def build_backbone(arch="convnext_small", in_chans=3):
    """Arch-generic Raptor backbone (random init). ViT-family: token-pool + dynamic img size.
    CNN (ConvNeXt/resnet/etc.): avg-pool. Multi-crop resolutions handled natively by conv."""
    is_vit = any(k in arch for k in ("vit", "deit", "eva", "beit", "swin"))
    kw = dict(pretrained=False, num_classes=0, in_chans=in_chans)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--out_dim", type=int, default=8192)
    ap.add_argument("--n_local", type=int, default=6)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--momentum", type=float, default=0.996)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.n_local, a.workers, a.warmup_epochs = 2, 8, 2, 0, 1

    aug = DinoAug(n_local=a.n_local)
    ds = KneeWindows(a.root, aug, limit=256 if a.smoke else None)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0,
                    prefetch_factor=(4 if a.workers > 0 else None))
    n_crops = 2 + a.n_local
    print(f"corpus {len(ds)} windows | bs {a.bs} | {len(dl)} steps/ep | crops {n_crops} | dev {dev}",
          flush=True)

    _bb = build_backbone(a.arch); feat_dim = _bb.num_features
    print(f"backbone {a.arch} | feat_dim {feat_dim}", flush=True)
    student = MultiCropWrapper(_bb, DINOHead(feat_dim, a.out_dim)).to(dev)
    teacher = MultiCropWrapper(build_backbone(a.arch), DINOHead(feat_dim, a.out_dim)).to(dev)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad = False

    dino = DINOLoss(a.out_dim, n_crops, teacher_temp=0.04, warmup_epochs=a.warmup_epochs,
                    n_epochs=a.epochs).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(dl)
    lr_s = cosine_sched(a.lr * a.bs / 256, 1e-6, a.epochs, steps, a.warmup_epochs, start=0)
    wd_s = cosine_sched(a.wd, 0.4, a.epochs, steps)
    mom_s = cosine_sched(a.momentum, 1.0, a.epochs, steps)

    start_ep = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev)
        student.load_state_dict(ck["student"]); teacher.load_state_dict(ck["teacher"])
        dino.load_state_dict(ck["dino"]); opt.load_state_dict(ck["opt"]); start_ep = ck["epoch"] + 1
        print(f"resumed from {a.resume} @ epoch {start_ep}", flush=True)

    gstep = start_ep * steps
    for ep in range(start_ep, a.epochs):
        student.train(); t0 = time.time(); run = 0.0
        for it, crops in enumerate(dl):
            for g in opt.param_groups:
                g["lr"] = lr_s[min(gstep, len(lr_s) - 1)]
                g["weight_decay"] = wd_s[min(gstep, len(wd_s) - 1)]
            crops = [c.to(dev, non_blocking=True) for c in crops]
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                with torch.no_grad():
                    t_out = teacher(crops[:2])          # globals only
                s_out = student(crops)
                loss = dino(s_out.float(), t_out.float(), ep)
            if not torch.isfinite(loss):
                print("non-finite loss, skip", flush=True); opt.zero_grad(); gstep += 1; continue
            opt.zero_grad(); loss.backward()
            if ep == 0:                                  # freeze last layer for first epoch
                for n, p in student.named_parameters():
                    if "last" in n and p.grad is not None: p.grad = None
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
            with torch.no_grad():                        # EMA teacher
                m = mom_s[min(gstep, len(mom_s) - 1)]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m).add_((1 - m) * ps.detach())
            run += loss.item(); gstep += 1
            if it % 50 == 0:
                print(f"ep{ep} it{it}/{steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}",
                      flush=True)
        ck = {"student": student.state_dict(), "teacher": teacher.state_dict(),
              "dino": dino.state_dict(), "opt": opt.state_dict(), "epoch": ep, "args": vars(a)}
        torch.save(ck, os.path.join(a.out, "raptor_ssl_last.pt"))
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            torch.save(ck, os.path.join(a.out, f"raptor_ssl_ep{ep+1}.pt"))
        print(f"== epoch {ep} done | mean loss {run/steps:.4f} | {time.time()-t0:.0f}s ==", flush=True)
    print("RAPTOR SSL PRETRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
