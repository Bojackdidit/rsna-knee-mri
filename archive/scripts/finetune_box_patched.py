#!/usr/bin/env python3
"""Fine-tune Raptor (our SSL knee encoder) on the 12-finding RSNA knee task.

Loads the Raptor ViT-S/16 backbone from an SSL checkpoint, attaches a per-diagnosis
attention-MIL head, and FULLY fine-tunes (trainable encoder) on LLM soft labels.
Gate = macro-AUC on the 45 in-corpus gold studies (held out of training).

Design mirrors the proven recipe (per-finding attention over the slice/window set,
weighted BCE, AdamW OneCycle) but with the encoder trainable instead of frozen features.

Runs bf16 on a single 4070Ti.

Controls / gate:
  * --ckpt raptor_ssl_last.pt   fine-tune the finished Raptor encoder
  * --ckpt none                 random-init ViT-S control (proves SSL pretrain adds value)
  * early ckpt (e.g. raptor_ssl_ep20.pt) validates the pipeline fast

Examples:
  python finetune_raptor.py --smoke                         # tiny correctness run
  python finetune_raptor.py --ckpt ckpt/raptor_ssl_ep20.pt --limit 800 --epochs 4   # pipeline-validate
  python finetune_raptor.py --ckpt ckpt/raptor_ssl_last.pt --epochs 12 --tag full   # real run
Outputs (per --tag): raptor_ft_<tag>.pt (best), raptor_gold_<tag>.npz (gold preds+truth+ids), raptor_ft_<tag>.json
"""
import math
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
AUG_SET = set()      # set from --aug; comma list of geo,erase,blur,noise,scanner
CENTRE_FRAC = 1.0   # set from --centre_frac; 1.0 = original full-range sampling
SORT_WINDOWS = False   # True only for order-sensitive heads (GRU)
SLICE_DILATE = 1    # set from --slice_dilate; window is [c-d, c, c+d]. d=1 is the original.
CROP_FRAC = 1.0     # set from --crop_frac; centre-crop of H,W before resize (joint magnification)
# The 44-slice corpus is FIVE SEQUENCE SLOTS concatenated, boundaries recovered from the mask's
# per-index validity: 12/10/8/6/8 = Sagittal-FS, Sagittal, Coronal-FS, Coronal, Axial (slot 3 is
# present in only 74% of studies). Inference opens EVERY DICOM header in EVERY series to sort
# slices, so fewer SERIES is the only lever on scoring time - and the arm must be trained on the
# same subset or it is a train/serve mismatch.
# ============================================================================
# CORPUS GEOMETRY - the single most expensive mistake this project has made.
# ============================================================================
# ⛔ 2026-09-13: TRAIN AND SERVE WERE RUNNING DIFFERENT GEOMETRIES, and had been for weeks.
#     _setup_box.sh:28   assert v.shape == (4407, 44, 336, 336)      <- what we TRAINED on
#     every serve kernel SLOTS 27/21/18/12/18, MAXS = 96             <- what we SERVE
# 44 slices at span 0.06-0.94 against 96 slices at span 0.02-0.98. A window is three physically
# ADJACENT CORPUS slices stacked as RGB, so at 44 that triplet spans 2.18x MORE anatomy than the
# triplet the kernel hands the model at test time - the third channel, the only 3D information the
# model has, is systematically wrong at inference. The serve volume also contains peripheral slices
# (2-6% and 94-98% of each series) the model has never seen, and the kernel's own comment says that
# is exactly where the collateral ligaments and the lateral meniscus sit.
# The board record is unambiguous: across TWELVE reads and EIGHT architectures, no Vast-box-trained
# arm has ever cleared 0.928, while the two v11-matched arms give 0.941.
#
# So FULL_SLOTS is no longer a hard-coded constant that silently describes the wrong corpus. It is
# DERIVED from the corpus actually on disk, and `--expect_depth` (default 96, what the kernel serves)
# makes an off-geometry run FAIL LOUDLY instead of burning credit and producing an unshippable arm.
GEOMETRY_BY_DEPTH = {
    # depth: (slot boundaries, builder, span, status)
    96: ([(0, 27), (27, 48), (48, 66), (66, 78), (78, 96)], "knee_corpus_v11.py", (0.02, 0.98),
         "SHIPPABLE - this is what every serve kernel builds"),
    64: ([(0, 18), (18, 32), (32, 44), (44, 52), (52, 64)], "knee_corpus_v4.py", (0.06, 0.94),
         "LEGACY - the span that took the board to 0.923; not what ships"),
    44: ([(0, 12), (12, 22), (22, 30), (30, 36), (36, 44)], "knee_corpus_v2.py", (0.06, 0.94),
         "OFF-GEOMETRY - trains 2.18x coarser than the kernel serves; arms cap out at 0.928"),
}
SERVE_DEPTH = 96          # MAXS in every kernel under variants/; change only if the kernel changes
FULL_SLOTS = GEOMETRY_BY_DEPTH[SERVE_DEPTH][0]


def set_geometry(depth, expect=SERVE_DEPTH, allow_mismatch=False):
    """Point FULL_SLOTS at the corpus that is actually on disk, and refuse a silent mismatch.

    Returns the slot list. Raises unless the depth matches `expect` or the caller has explicitly
    opted into an off-geometry run. Failing here costs seconds; failing to fail here has cost this
    project roughly three weeks of GPU and twelve board reads.
    """
    global FULL_SLOTS
    if depth not in GEOMETRY_BY_DEPTH:
        raise SystemExit(
            "[geometry] corpus depth %d is not a known geometry. Known: %s. Add it to "
            "GEOMETRY_BY_DEPTH with its builder and span before training on it."
            % (depth, sorted(GEOMETRY_BY_DEPTH)))
    FULL_SLOTS = GEOMETRY_BY_DEPTH[depth][0]
    slots, builder, span, status = GEOMETRY_BY_DEPTH[depth]
    print("[geometry] corpus depth %d | slots %s | builder %s | span %.2f-%.2f | %s"
          % (depth, [b - a for a, b in slots], builder, span[0], span[1], status), flush=True)
    if depth != expect and not allow_mismatch:
        raise SystemExit(
            "\n*** TRAIN/SERVE GEOMETRY MISMATCH - refusing to train. ***\n"
            "   corpus on disk : %d slices (%s, %s)\n"
            "   kernel serves  : %d slices (%s)\n"
            "   A 3-slice RGB window spans %.2fx more anatomy at train than at serve, and the arm\n"
            "   will not be shippable. Rebuild the corpus with %s (the sharded builder is at\n"
            "   audit_pull/knee-mri-fine-96-corpus-shard-0/), or pass --allow_geometry_mismatch if\n"
            "   you genuinely intend an off-geometry experiment and have said so in the log.\n"
            % (depth, builder, status, expect, GEOMETRY_BY_DEPTH[expect][1],
               float(expect) / float(depth), GEOMETRY_BY_DEPTH[expect][1]))
    return FULL_SLOTS


WANT_SLOTS = False  # set from --slot_emb; keeps the (x, y) collate contract when off


def slot_of(c):
    """Which sequence slot a window centre falls in."""
    for s, (a, b) in enumerate(FULL_SLOTS):
        if a <= c < b:
            return s
    return len(FULL_SLOTS) - 1

CORPUS_SLOTS = [(0, 12), (12, 20)]   # subset corpus: Sagittal-FS + Coronal-FS only, already 224
KEEP_SLOTS = None   # set from --slots, e.g. '0,2,4' keeps the three fat-suppressed series
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


# ------------------------------- data ----------------------------------------
def apply_aug(x):
    """Standard augmentation stack, each piece independently switchable via --aug.

    Ablating augmentation is the step an 11th-place competitor puts immediately after the baseline
    ("ablate through the standard image augmentation stacks ... Change one thing at a time"), and it
    is the one we had never touched: by default this trainer applied a +/-10% global gain and nothing
    else. No rotation, no scale, no translation, no erasing, no blur. Our arms plateau from about
    epoch 12-16 on 3,479 studies, which is what a bare stack looks like.

    NO FLIPS, ever. Laterality is signal in this task (medial vs lateral findings are distinct
    labels) and a horizontal flip relabels the knee. Everything here is either intensity-domain or a
    small rigid/affine warp that preserves left-right orientation.
    """
    K = x.shape[0]
    if "geo" in AUG_SET:
        # small affine: rotation +/-10 deg, scale +/-10%, translation +/-5%. One sample per window.
        ang = (torch.rand(K) - 0.5) * (2 * 10.0) * math.pi / 180.0
        sc = 1.0 + (torch.rand(K) - 0.5) * 0.20
        tx = (torch.rand(K) - 0.5) * 0.10
        ty = (torch.rand(K) - 0.5) * 0.10
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.zeros(K, 2, 3)
        theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if "blur" in AUG_SET and random.random() < 0.3:
        k = torch.tensor([1.0, 2.0, 1.0]); k = (k[:, None] * k[None, :]); k = k / k.sum()
        k = k.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])
    if "noise" in AUG_SET:
        x = (x + torch.randn_like(x) * (random.random() * 0.03)).clamp(0, 1)
    if "erase" in AUG_SET:
        # random erasing: 1-2 boxes per window, 2-12% of area each, filled with the window mean
        H, W = x.shape[-2:]
        for i in range(K):
            for _ in range(random.randint(1, 2)):
                if random.random() > 0.5:
                    continue
                a = random.uniform(0.02, 0.12) * H * W
                r = random.uniform(0.4, 2.5)
                h = int(round((a * r) ** 0.5)); w = int(round((a / r) ** 0.5))
                if h < 1 or w < 1 or h >= H or w >= W:
                    continue
                y0 = random.randint(0, H - h); x0 = random.randint(0, W - w)
                x[i, :, y0:y0 + h, x0:x0 + w] = x[i].mean()
    return x

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
        # A window is three PHYSICALLY ADJACENT slices of ONE series stacked as RGB. The old code
        # built `range(lo+1, hi)` contiguously between the first and last valid index, which ignores
        # both dropped slots and the boundaries BETWEEN slots: at 44 slices with 5 slots there are 4
        # boundaries, so centres at 11, 21, 29, 35 produced triples spanning two different series
        # (e.g. slices 10,11,12 = Sagittal-FS, Sagittal-FS, Sagittal). Roughly a fifth of every
        # window ever trained on mixed two acquisitions into one image.
        # A centre is admissible only if c-1, c and c+1 are all valid AND all inside the same slot.
        _vs = set(int(v) for v in valid)
        _d = int(getattr(self, "_D", 0)) or (int(valid.max()) + 1)
        _slots = [(a, b) for a, b in FULL_SLOTS if a < _d] or [(0, _d)]
        cs = [c for a, b in _slots for c in range(a + 1, min(b, _d) - 1)
              if (c - 1) in _vs and c in _vs and (c + 1) in _vs]
        if not cs:
            lo, hi = int(valid.min()), int(valid.max())
            cs = [max(1, min((lo + hi) // 2, max(_d - 2, 1)))]
        # Centre-concentrated sampling. Both branches below otherwise spread windows across the
        # WHOLE valid range (train shuffles all of it, eval linspaces all of it), which is the
        # losing arm of a published slice-geometry result for this competition:
        #   192px / no crop / 24 spread      0.7695 OOF
        #   288px / 140mm crop / 12 adjacent 0.7777
        #   288px / 140mm crop / centre-9    0.7815
        # ("slice geometry mattered more than resolution"). CENTRE_FRAC keeps only the middle
        # fraction of candidate centres, applied identically to train and eval so the two stay
        # consistent. 1.0 reproduces the previous behaviour exactly.
        if CENTRE_FRAC < 1.0 and len(cs) > 3:
            keep = max(3, int(round(len(cs) * CENTRE_FRAC)))
            mid = (len(cs) - keep) // 2
            cs = cs[mid:mid + keep]
        if self.train:
            reps = self.k // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            _bag = pool[:self.k]
            # A GRU reads the windows as an ORDERED sequence - a tear runs across consecutive
            # slices - so the random bag has to be put back into slice order or the recurrence is
            # modelling noise. Pooling heads are order-agnostic and are left exactly as they were.
            return sorted(_bag) if SORT_WINDOWS else _bag
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
                # FULL_SLOTS, not CORPUS_SLOTS: the latter is the 2-slot 224 subset corpus and has
                # only two entries, so `--slots 0,2,4` raised IndexError and `--slots 0,1` silently
                # selected the wrong slice ranges. The file's own comment above warned about this.
                _a, _b = FULL_SLOTS[_si]
                _keep[_a:min(_b, len(m))] = True
            m = (np.asarray(m) > 0) & _keep
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        cs = self._centers(valid)
        wins = np.empty((len(cs), 3, self.res, self.res), np.float32)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        for j, c in enumerate(cs):
            c = max(1, min(c, self._D - 2))
            _d = SLICE_DILATE
            _lo = max(0, c - _d); _hi = min(vol.shape[0] - 1, c + _d)
            # Dilated window: same THREE channels and therefore the same cost per forward, but a
            # far wider span of the stack. Nine dilated windows cover what twenty-seven tight ones
            # do, which is the only way to buy evidence without buying runtime.
            tri = np.stack([vol[_lo], vol[c], vol[_hi]], 0).astype(np.float32) / 255.0  # (3,H,W)
            t = torch.from_numpy(tri)
            # ROI crop: keep the middle CROP_FRAC of H,W so the joint fills more of the frame before
            # the resize. a published ladder put a 140mm physical crop at the centre of the only
            # geometry result anyone published, and this is the one lever on our pipeline still
            # outside the ~0.008 band where seven architectural levers all died.
            # Train jitters the crop centre by +/-4% for augmentation; eval takes the exact centre so
            # it matches what the inference kernel must do.
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
            if AUG_SET:
                x = apply_aug(x)
            if AUG_SCANNER:
                # Anti-scanner augmentation. Measured 2026-09-09: the model's twelve outputs
                # predict the acquiring scanner at 2.1x chance and per-scanner AUC spreads 0.048,
                # so it keys on equipment signature. These are the axes equipment actually varies
                # along - global gain alone (above) does not cover them. All INTENSITY-domain:
                # no geometry is touched, so laterality stays intact.
                x = x.pow(0.8 + random.random() * 0.45)                        # gamma
                c = 0.85 + random.random() * 0.30                              # contrast about the mean
                x = ((x - x.mean()) * c + x.mean()).clamp(0, 1)
                if random.random() < 0.7:                                      # smooth bias field
                    K, _, H, W = x.shape
                    lo = torch.rand(K, 1, 4, 4) * 0.5 + 0.75
                    field = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)
                    x = (x * field).clamp(0, 1)
                if random.random() < 0.5:
                    x = (x + torch.randn_like(x) * (random.random() * 0.02)).clamp(0, 1)
        if self.norm == "imagenet":       # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        if WANT_SLOTS:
            # slot id per window, from the SAME centres the windows were built from
            s = torch.tensor([slot_of(int(c)) for c in cs], dtype=torch.long)
            return x, y, s
        return x, y


AUG_SCANNER = False


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    if WANT_SLOTS:
        return xs, ys, torch.stack([b[2] for b in batch])   # (B,K) slot ids
    return xs, ys


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False, drop_path=0.0):
    is_vit = any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if drop_path and drop_path > 0:
        # Stochastic depth. The standard regulariser that makes a long schedule productive instead
        # of merely overfit; we have never passed it at all. timm accepts it on both CNN and ViT
        # families, so it is safe to set generically.
        kw["drop_path_rate"] = float(drop_path)
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
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2, n_slot=0, pool="attn"):
        super().__init__()
        self.backbone = backbone
        # How the per-window features become one vector per study.
        #   "attn"  per-finding softmax attention over windows - what every arm we have ever shipped
        #           uses, and what an 11th-place competitor explicitly lists as an EXPERIMENTAL thing
        #           to try LAST, after a simple baseline is built up.
        #   "mean" / "max"  plain pooling over windows, no learned weighting. He reports 0.94+ at
        #           224px with a ResNet34/EffNetB0 and simple pooling, which is both a different
        #           architecture AND roughly an order of magnitude cheaper per study than ours.
        # Kept as a flag rather than a rewrite so the two are measurable against each other on the
        # same fold, same labels, one variable.
        self.pool = pool
        self.norm = nn.LayerNorm(F_dim)
        # Per-sequence-slot embedding. Every finding currently reads the same globally-pooled vector
        # and the head has no idea whether a window came from the sagittal, coronal or axial stack -
        # so per-finding window attention cannot prefer the plane a finding is actually visible in.
        # The measured symptom is outputs that over-correlate (+0.168 median across all 66 pairs)
        # with the weakest findings being the localized ones (MCL, Synovitis, Lateral Meniscus).
        # ZERO-initialised, so at step 0 this is bit-identical to the model without it and the
        # optimiser buys plane-selectivity only where it pays.
        self.slot_emb = nn.Embedding(n_slot, F_dim) if n_slot else None
        if self.slot_emb is not None:
            nn.init.zeros_(self.slot_emb.weight)
        # Sequence head. The RSNA-2023 and RSNA-2022 first-place solutions both used this shape -
        # a 2.5D CNN encoder feeding a recurrent head over the ordered slice windows - while every
        # arm we have shipped uses order-agnostic pooling, which cannot represent a tear running
        # across consecutive slices at all. It is also architecturally unlike our existing stack,
        # so it decorrelates the blend rather than adding another near-twin.
        # GATED TO ZERO at init, so step 0 is bit-identical to pool="attn" and the optimiser buys
        # the recurrence only where it pays.
        self.gru = None
        if pool == "gru":
            self.gru = nn.GRU(F_dim, F_dim // 2, batch_first=True, bidirectional=True)
            self.gru_gate = nn.Parameter(torch.zeros(1))
            self.pool = "attn"
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

    def head(self, feats, slots=None):                    # feats: (B,K,F) -> (B,n)
        if self.slot_emb is not None and slots is not None:
            feats = feats + self.slot_emb(slots)           # (B,K,F), zero at init
        h = self.norm(feats)
        if self.gru is not None:
            _g, _ = self.gru(h)               # (B,K,F) over the ordered windows
            h = h + self.gru_gate * _g        # zero at init -> exact parity with attn
        if self.pool == "attn":
            a = self.att(h)                               # (B,K,n)
            a = torch.softmax(a, dim=1)                   # attention over windows, per finding
            pooled = torch.einsum("bkn,bkf->bnf", a, h)   # (B,n,F)
            return (pooled * self.clsW).sum(-1) + self.clsb
        # simple pooling: one vector per study, then one linear read-out per finding
        g = h.mean(1) if self.pool == "mean" else h.max(1).values      # (B,F)
        return g @ self.clsW.t() + self.clsb              # (B,n)

    def forward(self, x, slots=None):
        return self.head(self.encode(x), slots)


# ------------------------------- train ---------------------------------------
def make_loss(pw, mode="bce", rank_w=0.0):
    """Training objective.

    mode="bce"        the incumbent: one BCEWithLogits mean over every cell of the (B,12) matrix.
                      A finding whose soft labels sit almost entirely near 0 produces a small loss
                      AND a small gradient, so it is effectively down-weighted during training.
                      That is exactly the profile of our weakest findings - Synovitis is 84%
                      report-silent and Fracture 93% confident-zero.
    mode="macro_bce"  equalise the per-finding gradient magnitude, then average the twelve. Note
                      that naively taking l.mean(0).mean() is NOT this - for a full matrix it is
                      algebraically identical to l.mean(). Real macro weighting has to divide by
                      each finding's own detached scale.
    rank_w            weight on a pairwise ranking surrogate. BCE optimises calibration; the
                      competition scores ORDER only. Every (positive, negative) pair within the
                      batch contributes softplus(-(s_pos - s_neg)), per finding. Costs nothing at
                      inference.
    """
    bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    def _rank(logits, y):
        pos = (y > 0.5).float()
        neg = (y < 0.5).float()
        d = logits.unsqueeze(1) - logits.unsqueeze(0)        # (B,B,n): s_i - s_j
        m = pos.unsqueeze(1) * neg.unsqueeze(0)              # (B,B,n): i positive, j negative
        cnt = m.sum((0, 1))                                  # (n,) pairs available per finding
        per = (F.softplus(-d) * m).sum((0, 1)) / cnt.clamp(min=1.0)
        keep = cnt > 0
        if not bool(keep.any()):
            return logits.sum() * 0.0                        # keeps the graph, contributes nothing
        return per[keep].mean()

    def _loss(logits, y):
        # y is (B,12) normally, or (B,24) when a cell mask is supplied: the first 12 are the labels
        # and the last 12 are 1 = train on this cell, 0 = the report never addressed it.
        n = logits.shape[1]
        if y.shape[1] == 2 * n:
            y, cm = y[:, :n], y[:, n:]
        else:
            cm = None
        l = bce(logits, y)                                   # (B,n)
        if cm is not None:
            l = l * cm
        if mode == "macro_bce":
            if cm is not None:
                per_f = l.sum(0) / cm.sum(0).clamp(min=1.0)  # mean over the KEPT cells only
            else:
                per_f = l.mean(0)                            # (n,)
            w = (per_f.detach() + 1e-6).reciprocal()
            w = w / w.sum() * per_f.numel()                  # mean weight 1, so the scale is kept
            base = (per_f * w).mean()
        else:
            base = l.sum() / cm.sum().clamp(min=1.0) if cm is not None else l.mean()
        if rank_w > 0:
            base = base + rank_w * _rank(logits, y if cm is None else y * cm + 0.5 * (1 - cm))
        return base

    return _loss


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
    ap.add_argument("--llrd", type=float, default=0.0,
                    help="layer-wise lr decay per group toward the input; 0 disables")
    ap.add_argument("--llrd_groups", type=int, default=8)
    ap.add_argument("--pool", default="attn", choices=["attn", "mean", "max", "gru"],
                    help="how per-window features are pooled into one study vector; 'mean'/'max' drop "
                         "the attention head entirely (simple-baseline shape)")
    ap.add_argument("--slot_emb", action="store_true",
                    help="give the head a zero-initialised per-sequence-slot embedding so per-finding "
                         "window attention can prefer the plane a finding is visible in (parity at init)")
    ap.add_argument("--slice_dilate", type=int, default=1,
                    help="window = [c-d, c, c+d]; d=1 reproduces the original 3-adjacent-slice window")
    ap.add_argument("--init_arm", default="",
                    help="start from a FINETUNED arm (full RaptorClassifier state) instead of a bare "
                         "backbone, so an existing 0.941-class arm can be ADAPTED to a new window "
                         "geometry in a few epochs rather than retrained from scratch")
    ap.add_argument("--centre_frac", type=float, default=1.0,
                    help="keep only the middle fraction of candidate window centres")
    # Scanner-GROUPED validation (opt-in; default behaviour is unchanged).
    # Our 5 existing folds are 99% scanner-contaminated and the model's outputs predict the
    # scanner at 2.1x chance, so random-fold OOF is optimistic. Holding out whole scanner
    # fingerprints gives a gate that measures generalisation to unseen equipment instead.
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay (0.9998 per spec). Evaluates and SAVES the EMA weights; 0 = off")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="study-level mixup alpha (0.2 per spec); soft BCE targets mix linearly")
    ap.add_argument("--drop_path", type=float, default=0.0,
                    help="stochastic depth rate passed to the timm backbone (0.2 per spec)")
    ap.add_argument("--reclaim", type=float, default=0.0,
                    help="fraction of the held-out fold to give BACK to training, split by "
                         "fingerprint so no scanner appears on both sides (0.7 = Dread's 70%%)")
    ap.add_argument("--aug", default="",
                    help="comma list of augmentations to enable: geo,erase,blur,noise. Default is "
                         "NONE, which is what every arm so far was trained with.")
    ap.add_argument("--aug_scanner", action="store_true",
                    help="anti-scanner intensity augmentation: gamma, contrast, bias field, noise")
    ap.add_argument("--group_csv", default="",
                    help="study_fingerprint.csv: StudyInstanceUID,fp -> enables grouped validation")
    ap.add_argument("--group_fold", type=int, default=0, help="which of --group_folds to hold out")
    ap.add_argument("--group_folds", type=int, default=5)
    ap.add_argument("--loss", default="bce", choices=["bce", "macro_bce"],
                    help="macro_bce equalises each finding's gradient magnitude. The metric is a "
                         "macro mean; the loss never was.")
    ap.add_argument("--cell_mask", default="",
                    help="parquet of 0/1 per (study,finding). A 0 means the radiology report never "
                         "ADDRESSED that finding, so the label carries no information and the cell "
                         "is excluded from the loss. Our three weakest board findings are the three "
                         "most report-silent ones (Synovitis 84%% silent, Lateral OA 33%%, PF OA "
                         "18.5%%; they average 0.878 against 0.928 for the clean ones), and the "
                         "incumbent loss trains on every one of those empty cells.")
    ap.add_argument("--rank_w", type=float, default=0.0,
                    help="weight on the pairwise ranking surrogate (0.05 per the public resgated "
                         "trainer). BCE optimises calibration, the board scores order.")
    ap.add_argument("--expect_depth", type=int, default=SERVE_DEPTH,
                    help="slice depth the SERVING kernel builds (MAXS). The corpus on disk must "
                         "match it or the run is refused - see GEOMETRY_BY_DEPTH at the top.")
    ap.add_argument("--allow_geometry_mismatch", action="store_true",
                    help="deliberately train off-geometry. Only for an experiment you have written "
                         "down; the resulting arm is NOT shippable by the current kernel.")
    a = ap.parse_args()
    global AUG_SCANNER
    AUG_SCANNER = bool(a.aug_scanner)   # before any worker forks
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag}", flush=True)

    # ---- geometry gate: FIRST, before a model or a dataloader exists ----
    # Costs one memmap header read. Catches the mismatch that capped every box-trained arm at 0.928.
    _gdepth = int(np.load(os.path.join(DATA, "all_vols.npy"), mmap_mode="r").shape[1])
    set_geometry(_gdepth, expect=a.expect_depth, allow_mismatch=a.allow_geometry_mismatch)

    # ---- ids / labels ----
    ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    tr = pd.read_csv(os.path.join(RSNA, "train.csv")); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    # Which soft-label set to train on. Our BEST live arm (confdist_swa, gold 0.9266) was fitted
    # on labels_confdist.parquet, while the default here is the plain LLM set - which is why a
    # fresh 8-epoch run reproduced only 0.8785. All sets are drop-in: 4349 rows, same 13 columns.
    _lab_path = a.labels if os.path.isabs(a.labels) else os.path.join(RSNA, "labeler", a.labels)
    print(f"[labels] {_lab_path}", flush=True)
    soft = pd.read_parquet(_lab_path)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    group_val_ids = []
    if a.group_csv:
        gdf = pd.read_csv(a.group_csv)
        gdf["StudyInstanceUID"] = gdf["StudyInstanceUID"].astype(str)
        fp_of = dict(zip(gdf.StudyInstanceUID, gdf.fp.astype(str)))
        # greedy balanced assignment: biggest fingerprints first into the lightest fold, so no
        # fingerprint is ever split across the train/val boundary
        sizes = {}
        for u in train_ids:
            f = fp_of.get(u)
            if f is not None:
                sizes[f] = sizes.get(f, 0) + 1
        load = [0] * a.group_folds
        fold_of_fp = {}
        for f, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
            j = int(np.argmin(load)); fold_of_fp[f] = j; load[j] += n
        held = {f for f, j in fold_of_fp.items() if j == a.group_fold}
        group_val_ids = [u for u in train_ids if fp_of.get(u) in held]
        train_ids = [u for u in train_ids if fp_of.get(u) not in held]
        if a.reclaim > 0.0:
            # Give most of the held-out fold BACK to training and validate on what is left.
            # The shipped champion arms train on all 4,349 studies with no holdout at all; ours
            # surrender 870 to have a gate, and they measure 0.009 behind at matched window count.
            # This recovers most of that data while keeping a real, scanner-disjoint validation set.
            # The split is by FINGERPRINT, not by study: splitting inside a fingerprint would put the
            # same scanner on both sides and hand the validation set back its own training data.
            _hf = sorted(held)
            # FIXED split seed, deliberately not a.seed: arms in a pair are trained with different
            # seeds, and if the reclaim split followed the training seed each arm would hold out a
            # DIFFERENT set - so the pair could never be scored on common studies, which is the only
            # way a pair has ever been evaluated here.
            _rng = np.random.default_rng(20260912)
            _rng.shuffle(_hf)
            # Take whole fingerprints until 70% of the held-out STUDIES are covered. Taking 70% of
            # the fingerprints instead returns a variable and much smaller share of studies - here it
            # gave 362 of 870 (42%) - because fingerprints differ wildly in size.
            _size = {}
            for _u in group_val_ids:
                _f = fp_of.get(_u)
                _size[_f] = _size.get(_f, 0) + 1
            # Take whole fingerprints, but NEVER let the validation set fall below (1 - reclaim) of
            # the fold. Fingerprint sizes here are wildly skewed - 350, 118, 94, ... down to 1 - so
            # simply accumulating until a target is crossed overshoots catastrophically: it returned
            # 867 of 870 studies and left THREE to validate on. Skipping any fingerprint that would
            # breach the floor yields 597 returned (69%) and 273 held back (31%).
            _tot = sum(_size.values())
            _floor = (1.0 - a.reclaim) * _tot
            _back, _acc = set(), 0
            for _f in _hf:
                if (_tot - _acc - _size.get(_f, 0)) < _floor:
                    continue
                _back.add(_f); _acc += _size.get(_f, 0)
            _keep = set(_hf) - _back
            _moved = [u for u in group_val_ids if fp_of.get(u) in _back]
            group_val_ids = [u for u in group_val_ids if fp_of.get(u) in _keep]
            train_ids = train_ids + _moved
            print("[reclaim] %.0f%% of the held-out fold returned to training: "
                  "%d of %d fingerprints, %d studies -> train %d / validate %d"
                  % (100 * a.reclaim, len(_back), len(_hf), len(_moved),
                     len(train_ids), len(group_val_ids)), flush=True)
            assert not (set(fp_of.get(u) for u in train_ids) & _keep), "scanner leaked into validation"
        print(f"[grouped] {len(sizes)} fingerprints -> {a.group_folds} folds, sizes {load}", flush=True)
        print(f"[grouped] holding out fold {a.group_fold}: {len(held)} fingerprints, "
              f"{len(group_val_ids)} studies; train {len(train_ids)}", flush=True)
        # Check against the fingerprints STILL being validated on, not the original held set:
        # --reclaim deliberately moves some of the held fingerprints back into training, so testing
        # against `held` fires by construction. What must remain true is that nothing in training
        # shares a fingerprint with anything in the validation set.
        _valfp = {fp_of.get(u) for u in group_val_ids}
        assert not ({fp_of.get(u) for u in train_ids} & _valfp), "scanner leaked across the split"
        # Record exactly which studies this run validated on. Runs with a different --reclaim (or
        # a different --group_fold) hold out DIFFERENT sets, so comparing their reported bests is
        # meaningless: E0 validates on 350 studies and E1 on all 870 of the same fold, and E0 has
        # trained on 520 of E1's. The only honest comparison is the INTERSECTION of the two val
        # sets, and computing that needs this file. Cheap to write, impossible to reconstruct once
        # the box is gone.
        try:
            with open("valids_%s.csv" % a.tag, "w") as _vf:
                _vf.write("StudyInstanceUID\n")
                for _u in group_val_ids:
                    _vf.write("%s\n" % _u)
            print("[grouped] wrote valids_%s.csv (%d studies)" % (a.tag, len(group_val_ids)), flush=True)
        except Exception as _e:
            print("[grouped] could not write val id list: %s" % _e, flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in group_val_ids:
        labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}", flush=True)

    # pos_weight from soft-label prevalence (imbalance handling)
    # ---- cell mask -----------------------------------------------------------------------
    # Appended to each label vector so the Dataset, collate, mixup and slot paths are untouched:
    # y becomes (24,) = 12 labels then 12 keep-flags, and make_loss splits it. evaluate() only ever
    # indexes the first 12, so validation is unaffected by construction.
    if a.cell_mask:
        _cm = pd.read_parquet(a.cell_mask)
        _cm["StudyInstanceUID"] = _cm["StudyInstanceUID"].astype(str)
        _cm = _cm.set_index("StudyInstanceUID")
        _keep_n = 0
        for _u in list(labels):
            if _u in _cm.index:
                _m = _cm.loc[_u, LAB].values.astype(np.float32)
            else:
                _m = np.ones(len(LAB), np.float32)
            if _u in goldset:
                _m = np.ones(len(LAB), np.float32)   # gold labels are real; never mask them
            _keep_n += int(_m.sum())
            labels[_u] = np.concatenate([labels[_u], _m])
        print("[cell_mask] %s | training on %.1f%% of label cells; the rest were never addressed "
              "by the report" % (os.path.basename(a.cell_mask),
                                 100.0 * _keep_n / (len(labels) * len(LAB))), flush=True)

    _lab_only = np.stack([labels[u][:len(LAB)] for u in train_ids])
    prev = np.clip(_lab_only.mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    tds = StudyWindows(DATA, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm)
    val_ids = group_val_ids if a.group_csv else gold_ids
    if a.group_csv:
        print(f"[grouped] validating on {len(val_ids)} scanner-disjoint studies "
              f"(soft labels), NOT the retired gold-58 gate", flush=True)
    vds = StudyWindows(DATA, val_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)

    global CENTRE_FRAC, CROP_FRAC, SLICE_DILATE, SORT_WINDOWS
    SLICE_DILATE = int(a.slice_dilate)
    print(f'[window] slices [c-{SLICE_DILATE}, c, c+{SLICE_DILATE}]', flush=True)
    CENTRE_FRAC = float(a.centre_frac)
    # A recurrent head only means anything if the windows arrive in slice order, and the training
    # sampler shuffles its bag. Switch that off for GRU runs only.
    SORT_WINDOWS = (a.pool == "gru")
    print("[pool] %s%s" % (a.pool, " (windows sorted into slice order)" if SORT_WINDOWS else ""), flush=True)
    CROP_FRAC = float(a.crop_frac)
    global KEEP_SLOTS
    global AUG_SET
    AUG_SET = {t.strip() for t in a.aug.split(",") if t.strip()}
    print("[aug] %s" % (sorted(AUG_SET) or "none (bare +/-10% gain only)"), flush=True)
    # AUG_SCANNER was silently absent from this line, so a run could not be told apart from one
    # without it except by its loss curve. Print it.
    print("[aug_scanner] %s" % ("ON - gamma, contrast, bias field, fine noise" if AUG_SCANNER else "off"), flush=True)
    KEEP_SLOTS = [int(x) for x in a.slots.split(',') if x.strip() != ''] or None
    print(f'[slots] keeping corpus slots {KEEP_SLOTS if KEEP_SLOTS else "ALL"}', flush=True)
    print(f'[crop] centre crop {CROP_FRAC:.2f} of H,W before resize', flush=True)
    print(f"[centre] window centres restricted to the middle {CENTRE_FRAC:.2f} of the stack", flush=True)
    use_timm = a.ckpt in ("timm", "pretrained")
    bb = build_backbone(a.arch, pretrained=use_timm, drop_path=a.drop_path)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    global WANT_SLOTS
    WANT_SLOTS = bool(a.slot_emb)
    _nslot = len(FULL_SLOTS) if WANT_SLOTS else 0
    if WANT_SLOTS:
        print("[slots] per-slot embedding ON (%d slots, zero-init: parity with the plain head at step 0)"
              % _nslot, flush=True)
    print("[head] pooling = %s" % a.pool, flush=True)
    model = RaptorClassifier(bb, F_dim=F_dim, n_slot=_nslot, pool=a.pool).to(dev)
    if a.init_arm:
        _ck = torch.load(a.init_arm, map_location='cpu', weights_only=False)
        _miss, _unexp = model.load_state_dict(_ck['model'], strict=False)
        print(f'[init_arm] {os.path.basename(a.init_arm)} | missing {len(_miss)} unexpected {len(_unexp)} '
              f'| src gold {_ck.get("gold_auc")}', flush=True)
    if a.freeze_blocks > 0:
        for nm, p in model.backbone.named_parameters():
            for b in range(a.freeze_blocks):
                if nm.startswith(f"blocks.{b}."): p.requires_grad = False

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    # LAYER-WISE LR DECAY. The shipped arms sit 0.0220 from their ImageNet init while the head moves
    # 0.8953 - we have been training a 0.28M head on a frozen extractor. Raising the backbone rate
    # was tried and failed (5e-5 -0.003, 1e-4 -0.014, 3e-4 diverges), but every one of those runs
    # used ONE flat rate across all 72.9M parameters, so the early layers - which should barely move
    # - got the same rate as the late ones that need to adapt. LLRD is the standard resolution:
    # bb_lr applies to the TOP of the backbone and decays geometrically toward the input.
    if a.llrd and a.llrd < 1.0:
        _named = [(n_, p_) for n_, p_ in model.backbone.named_parameters() if p_.requires_grad]
        _G = max(1, int(a.llrd_groups))
        _per = max(1, (len(_named) + _G - 1) // _G)
        _groups = []
        for _gi in range(_G):
            _chunk = [p_ for _, p_ in _named[_gi * _per:(_gi + 1) * _per]]
            if not _chunk:
                continue
            # group 0 is nearest the INPUT, so it gets the most decay
            _scale = a.llrd ** (_G - 1 - _gi)
            _groups.append({"params": _chunk, "lr": a.bb_lr * _scale})
        _groups.append({"params": head_params, "lr": a.head_lr})
        opt = torch.optim.AdamW(_groups, weight_decay=a.wd)
        print("[llrd] decay %.2f over %d groups | backbone lr %.2e (top) -> %.2e (input) | head %.2e"
              % (a.llrd, len(_groups) - 1, a.bb_lr, a.bb_lr * (a.llrd ** (len(_groups) - 2)),
                 a.head_lr), flush=True)
    else:
        opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                                 {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(len(tl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps, pct_start=0.15)
    lossf = make_loss(pw, mode=a.loss, rank_w=a.rank_w)
    print("[loss] %s%s" % (a.loss, " + %.3f * pairwise-rank" % a.rank_w
                           if a.rank_w > 0 else ""), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval(); P = []; Y = []
        for _b in vl:
            x, y, s = (_b if len(_b) == 3 else (_b[0], _b[1], None))
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev), None if s is None else s.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            # Gold labels are already 0/1, but the scanner-grouped validation set carries SOFT
            # labels in [0,1]; .astype(int) truncated those to all-zero, every class check failed
            # and the macro mean came out NaN. Threshold at 0.5 - a no-op for the binary gold path.
            yb = (Y[:, j] >= 0.5).astype(int)
            if len(set(yb.tolist())) > 1:
                aucs[name] = float(roc_auc_score(yb, P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    # EMA of the weights. A long schedule spends its last epochs oscillating around the optimum;
    # the running average sits closer to it than any single step does, and it costs one extra copy
    # of the weights and one lerp per step. We evaluate AND save the EMA weights when it is on.
    ema_model = None
    if a.ema and a.ema > 0:
        import copy as _copy
        ema_model = _copy.deepcopy(model).eval()
        for _p in ema_model.parameters():
            _p.requires_grad_(False)
        print("[ema] decay %.5f - evaluating and saving the EMA weights" % a.ema, flush=True)

    def _ema_update():
        d = a.ema
        with torch.no_grad():
            for _e, _m in zip(ema_model.state_dict().values(), model.state_dict().values()):
                if _e.dtype.is_floating_point:
                    _e.mul_(d).add_(_m.detach(), alpha=1.0 - d)
                else:
                    _e.copy_(_m)

    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for _b in tl:
            x, y, s = (_b if len(_b) == 3 else (_b[0], _b[1], None))
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if a.mixup and a.mixup > 0 and x.shape[0] > 1:
                # Study-level mixup: blend whole studies and their SOFT label vectors by the same
                # lambda. BCE takes soft targets natively, so no loss change is needed. Slot ids are
                # left untouched - they index the sequence, not the pixels.
                _lam = float(np.random.beta(a.mixup, a.mixup))
                _perm = torch.randperm(x.shape[0], device=x.device)
                x = _lam * x + (1.0 - _lam) * x[_perm]
                y = _lam * y + (1.0 - _lam) * y[_perm]
            opt.zero_grad()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x, None if s is None else s.to(dev))
                loss = lossf(logits.float(), y)   # composite: see make_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); sched.step(); tot += loss.item()
            if ema_model is not None:
                _ema_update()
        if ema_model is not None:
            _live = model
            model = ema_model
            au, aucs, P, Y = evaluate()
            model = _live
        else:
            au, aucs, P, Y = evaluate()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if au > best:
            best = au; best_P = P
            _src_model = ema_model if ema_model is not None else model
            # Record the GEOMETRY too. StudyWindows reads module globals, so anything scoring this
            # arm later has to restore them - a 3-series arm evaluated on the full 5-series volume
            # reads systematically low. Without this the checkpoint cannot say what it needs.
            best_state = {"model": {k: v.detach().cpu() for k, v in _src_model.state_dict().items()},
                          "slots": list(KEEP_SLOTS) if KEEP_SLOTS else None,
                          "centre_frac": CENTRE_FRAC, "crop_frac": CROP_FRAC,
                          "slice_dilate": SLICE_DILATE, "pool": a.pool,
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                          "arch": a.arch, "lab": LAB, "epoch": ep}
            # Write the checkpoint AS SOON AS it improves, not only after the final epoch. A 5-hour
            # CoAtNet run held its only copy in memory until the loop ended, so a box dying at
            # epoch 15 of 16 would have lost everything - and it also made the arm impossible to
            # measure mid-flight when a decision depended on it. Atomic rename so a kill during the
            # write cannot leave a truncated file.
            try:
                _tmp = os.path.join(HERE, "raptor_ft_%s.pt.partial" % a.tag)
                torch.save(best_state, _tmp)
                os.replace(_tmp, os.path.join(HERE, "raptor_ft_%s.pt" % a.tag))
            except Exception as _se:
                print("[ckpt] could not write early checkpoint: %s" % _se, flush=True)
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
                 pred=best_P, truth=Y[:, :len(LAB)], ids=np.array(val_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids)},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
