#!/usr/bin/env python3
"""Throughput benchmark on the REAL training step, synthetic data.

Mirrors finetune_raptor_cnn5.py exactly: coatnet_rmlp_2_rw_384 + the attention-MIL head,
bf16 autocast, AdamW, bs 8 studies x k 12 windows at 384px. Synthetic tensors so no corpus
transfer is needed - this isolates GPU throughput, which is what we are choosing between.

Baseline to beat, measured on the RTX A6000 from a real run:
    3155 studies x 12 windows / 840 s per epoch = 45.1 windows/s, GPU pinned at 100%.
"""
import time, argparse, torch, torch.nn as nn, timm

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
ap.add_argument("--res", type=int, default=384)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--warmup", type=int, default=8)
a = ap.parse_args()

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim)); self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
    def forward(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, K, -1)
        h = self.norm(f)
        w = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", w, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

dev = "cuda"
print(f"gpu: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
res = {}
for ckpt in (True, False):
    try:
        bb = timm.create_model(a.arch, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
        if ckpt: bb.set_grad_checkpointing(True)
        m = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.02)
        lossf = nn.BCEWithLogitsLoss()
        x = torch.randn(a.bs, a.k, 3, a.res, a.res, device=dev)
        y = torch.randint(0, 2, (a.bs, 12), device=dev).float()
        torch.cuda.reset_peak_memory_stats(); t0 = None
        for i in range(a.warmup + a.steps):
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = lossf(m(x), y)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            if i == a.warmup - 1:
                torch.cuda.synchronize(); t0 = time.time()
        torch.cuda.synchronize()
        dt = time.time() - t0
        wps = a.steps * a.bs * a.k / dt
        peak = torch.cuda.max_memory_allocated() / 2**30
        res[ckpt] = (wps, peak)
        print(f"grad_ckpt={'ON ' if ckpt else 'OFF'} | {dt/a.steps*1000:7.1f} ms/step | "
              f"{wps:6.1f} windows/s | peak {peak:5.1f} GiB | vs A6000 45.1 w/s = {wps/45.1:.2f}x", flush=True)
        del m, opt, x, y; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"grad_ckpt={'ON ' if ckpt else 'OFF'} | FAILED: {str(e)[:90]}", flush=True)
        torch.cuda.empty_cache()
if res:
    best = max(v[0] for v in res.values())
    print(f"\nBEST {best:.1f} windows/s = {best/45.1:.2f}x the A6000")
    print(f"16-epoch run (37,860 windows/epoch): A6000 3.73 h -> this GPU {37860*16/best/3600:.2f} h")
