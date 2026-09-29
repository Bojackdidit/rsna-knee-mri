#!/usr/bin/env python3
"""Add a finding-specific SPATIAL gate to the arm. Written to start EXACTLY at the current model.

Where the gap is. Our head already pools across WINDOWS per finding (attention-MIL), which is the
second half of what the frontier does. What it does not have is spatial selectivity INSIDE each
image: the backbone is built with global_pool="avg", so every 384x384 knee is collapsed to one
vector before the head ever sees it. For small localized findings - meniscus, ACL, fracture - that
averages the lesion away across the whole joint. Our own weak findings are exactly the localized
ones, and our notes have said for weeks that the frontier shape is "anatomy-localized ROI and
non-global pooling" (tomoon33 won RSNA 2025 with nnU-Net -> ROI -> classifier).

What this adds, per finding n:
    smap_n = Conv1x1(feature_map)            per-finding spatial logits over the h x w grid
    loc_n  = sum_p softmax(smap_n)_p * fmap_p   finding-specific spatial pool
    feat_n = gap + g_n * (loc_n - gap)       gap is the protected global-average residual

g is a free per-finding parameter initialised to ZERO, so at step 0 feat_n == gap for every finding
and the model is bit-identical to the current one; the optimiser can then buy spatial selectivity
only where it pays. Attention over windows is still computed from the global vector, so that path
is unchanged and stable.

No ROI detector, no extra labels, no inference-time cost beyond one 1x1 conv.
Enabled with --spatial_gate; without the flag the file behaves exactly like cnn5.
"""
import re, sys

src = open(sys.argv[1], encoding="utf-8").read()

# 1. backbone must hand back a feature MAP when the gate is on
old_bb = '''def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    is_vit = any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)'''
new_bb = '''def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    is_vit = any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)'''
assert old_bb in src

# 2. the head
old_cls = src[src.index("class RaptorClassifier(nn.Module):"):src.index("    def forward(self, x):\n        return self.head(self.encode(x))") + len("    def forward(self, x):\n        return self.head(self.encode(x))")]
new_cls = '''class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings).

    With spatial_gate=True the classifier reads a FINDING-SPECIFIC spatial pool of each window
    instead of the window's global average. The gate is initialised to zero, so the model starts
    numerically identical to the ungated one and can only depart from it if that lowers the loss.
    """
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2, spatial_gate=False):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))       # per-window, per-finding attention score
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n
        self.spatial_gate = spatial_gate
        if spatial_gate:
            self.smap = nn.Conv2d(F_dim, n, kernel_size=1)
            nn.init.zeros_(self.smap.weight); nn.init.zeros_(self.smap.bias)
            # g = 0 -> feat_n == global average -> identical to the ungated model at init
            self.gate = nn.Parameter(torch.zeros(n))

    def _featmap(self, xf):
        """(B*K,3,H,W) -> (B*K,C,h,w) as NCHW, whatever layout the backbone emits."""
        fm = self.backbone.forward_features(xf)
        if fm.dim() == 3:                                  # (B,T,C) token grid -> square map
            B, T, C = fm.shape
            s = int(round(T ** 0.5))
            if s * s != T:                                 # drop leading CLS/prefix tokens
                s = int(round((T - 1) ** 0.5)); fm = fm[:, T - s * s:, :]
            fm = fm.transpose(1, 2).reshape(fm.shape[0], C, s, s)
        elif fm.dim() == 4 and fm.shape[-1] == self.norm.normalized_shape[0] and fm.shape[1] != self.norm.normalized_shape[0]:
            fm = fm.permute(0, 3, 1, 2).contiguous()       # NHWC -> NCHW
        return fm

    def encode(self, x):                                  # x: (B,K,3,H,W) -> (B,K,F) or (B,K,n,F)
        B, K = x.shape[:2]
        xf = x.flatten(0, 1)
        if not self.spatial_gate:
            return self.backbone(xf).view(B, K, -1)
        fm = self._featmap(xf)                            # (BK,C,h,w)
        BK, C = fm.shape[0], fm.shape[1]
        flat = fm.flatten(2)                              # (BK,C,P)
        gap = flat.mean(-1)                               # (BK,C) -- the protected residual
        aw = torch.softmax(self.smap(fm).flatten(2), dim=-1)          # (BK,n,P)
        loc = torch.einsum("bnp,bcp->bnc", aw, flat)                  # (BK,n,C)
        g = self.gate.view(1, -1, 1)
        feat = gap.unsqueeze(1) + g * (loc - gap.unsqueeze(1))        # (BK,n,C)
        return feat.view(B, K, self.n, C)

    def head(self, feats):
        if feats.dim() == 3:                              # (B,K,F) -- original path
            h = self.norm(feats)
            a = torch.softmax(self.att(h), dim=1)         # (B,K,n) over windows
            pooled = torch.einsum("bkn,bkf->bnf", a, h)   # (B,n,F)
        else:                                             # (B,K,n,F) -- spatial-gated path
            h = self.norm(feats)                          # (B,K,n,F)
            gapf = h.mean(2)                              # window-level vector for the attention
            a = torch.softmax(self.att(gapf), dim=1)      # (B,K,n) -- unchanged attention path
            pooled = torch.einsum("bkn,bknf->bnf", a, h)  # (B,n,F)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))'''
src = src.replace(old_cls, new_cls)

# 3. CLI flag + wire it through
src = src.replace('ap.add_argument("--head_lr", type=float, default=1e-3)',
                  'ap.add_argument("--head_lr", type=float, default=1e-3)\n'
                  '    ap.add_argument("--spatial_gate", action="store_true",\n'
                  '                    help="finding-specific spatial pooling; gate inits to 0 = identical to off")')
n_ctor = src.count("RaptorClassifier(bb, F_dim=F_dim)")
assert n_ctor == 1, "constructor call matched %d times" % n_ctor
src = src.replace("RaptorClassifier(bb, F_dim=F_dim)",
                  "RaptorClassifier(bb, F_dim=F_dim, spatial_gate=a.spatial_gate)")
open(sys.argv[2], "w", encoding="utf-8").write(src)
print("wrote", sys.argv[2], "| head replaced, --spatial_gate wired")
