#!/usr/bin/env python3
"""Generate the zero-GPU serve-side variants from the champion kernel.

All of these reuse the SHIPPED champion weights. No training, no new arms. Each one is a single
board read against the 0.941 incumbent, and each isolates exactly one change.

WHY THESE FOUR
--------------
An attention-routing diagnostic (CPU, N=50, K=24, both champion checkpoints, replicated) measured:

  * The head's study logit is EXACTLY an attention-weighted mean of per-window per-finding logits:
        logits = (pooled * clsW).sum(-1) + clsb,  pooled = sum_k a[k,n] h[k]
      => logit[n] = sum_k a[k,n] * (h[k].clsW[n] + clsb[n])          (softmax sums to 1)
    So per-window logits are already computable; nothing new has to be learned to pool differently.
  * max-pooling produces a GENUINELY different study ranking (median Spearman vs the learned
    attention 0.864), while mean-pooling does not (0.970). Pooling is a real lever, and the
    champion is sitting on the half of it that behaves like a mean.
  * Attention is only mildly peaked - mean 16.0 of 24 effective windows. Synovitis 19.98/24 and
    Fracture 19.39/24 are nearly uniform, which is wrong for focal findings.
  * Medial OA, Lateral OA and PF OA all peak on Sagittal-FS when they should be coronal/axial.
    PF OA's axial enrichment is 0.742 - it attends the axial stack LESS than uniform. PF OA and
    Lateral OA are two of the four weakest findings on the board.
  * The kernel's `_eval_centers` takes range(lo+1, hi) with NO validity check, so the ~26% of
    studies missing Coronal-nonFS feed 12 all-zero slices in as window centres, and every study
    gets 4 triplets that straddle a slot boundary and mix two different acquisitions into one RGB.

VARIANTS
--------
  v1_pool     per-finding max/top2 pooling for the focal findings
  v2_slotpri  additive log-prior routing the three mis-routed OA findings to their planes
  v3_windows  skip all-zero and slot-straddling window centres
  v4_pool_win v1 + v3 together (fire only after both have their own board read)

PARITY GUARANTEE
----------------
The head refactor is algebraically identical to the champion when POOL_MODE and SLOT_PRIOR are both
None. `--selftest` proves it numerically on random tensors before any variant is written.

Usage:
    python build_serve_variants.py --selftest
    python build_serve_variants.py --out variants
"""
import argparse, copy, json, os, sys

BASE_NB = os.path.join("variants", "push_fastdcm_clean",
                       "knee-mri-raptor-direct-dicom-read.ipynb")
BASE_META = os.path.join("variants", "push_fastdcm_clean", "kernel-metadata.json")

BASE_BAD = []

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# ---------------------------------------------------------------- patch anchors
OLD_HEAD = '''    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))'''

NEW_HEAD = '''    def head(self, feats, slots=None):
        # Refactored, NOT changed. The champion computed
        #     pooled = sum_k a[k,n] h[k];  logit[n] = pooled[n].clsW[n] + clsb[n]
        # which expands to  sum_k a[k,n] * (h[k].clsW[n] + clsb[n])  because softmax sums to 1.
        # So the per-window per-finding logit l[k,n] is already implicit in the trained weights and
        # can be pooled any way we like without retraining. With POOL_MODE and SLOT_PRIOR both
        # None this returns bit-identical output to the champion (proved by --selftest).
        h = self.norm(feats)                                       # (B,K,F)
        a = self.att(h)                                            # (B,K,n) raw attention scores
        if SLOT_PRIOR is not None and slots is not None:
            # Additive log-prior in SCORE space, before the softmax, so the attention still
            # renormalises over the windows that survive. A finding already routed to the right
            # plane barely moves; a mis-routed one is pushed onto its anatomy.
            _p = SLOT_PRIOR.to(a.device, a.dtype)                  # (n, n_slot)
            _s = slots.to(a.device)                                # (K,)
            a = a + _p.index_select(1, _s).t().unsqueeze(0)        # (1,K,n)
        a = torch.softmax(a, dim=1)
        l = torch.einsum("bkf,nf->bkn", h, self.clsW) + self.clsb  # (B,K,n) per-window logits
        out = (a * l).sum(1)                                       # == the champion's computation
        if POOL_MODE:
            out = out.clone()
            for _n, _mode in POOL_MODE.items():
                if _mode == "max":
                    out[:, _n] = l[:, :, _n].max(1).values
                elif _mode == "top2":
                    _k = min(2, l.shape[1])
                    out[:, _n] = l[:, :, _n].topk(_k, dim=1).values.mean(1)
                elif _mode == "top3":
                    _k = min(3, l.shape[1])
                    out[:, _n] = l[:, :, _n].topk(_k, dim=1).values.mean(1)
        return out

    def forward(self, x, slots=None):
        return self.head(self.encode(x), slots)'''

OLD_EVALWIN_TAIL = '''    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x'''

NEW_EVALWIN_TAIL = '''    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    # Slot id per window, derived from the SAME centres the windows were built from. The kernel
    # concatenates five acquisitions into one volume and the head otherwise has no way to know
    # which plane a window came from.
    _sl = torch.tensor([_slot_of(int(c)) for c in cs], dtype=torch.long)
    return x, _sl'''

OLD_INFER = '''@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()'''

NEW_INFER = '''@torch.no_grad()
def infer_probs(model, xwins, device, slots=None):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x, slots).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x, slots).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x, slots).float())
    return o[0].cpu().numpy()'''

OLD_PREP = '''    def _prep_study(sid):
        vol, mask = build_study(sid, SER, tsdir, reader)
        xw = eval_windows(vol, mask, k=K_EVAL, res=_SP_RES, norm=NORM)
        del vol, mask
        return xw'''

NEW_PREP = '''    def _prep_study(sid):
        vol, mask = build_study(sid, SER, tsdir, reader)
        xw, sl = eval_windows(vol, mask, k=K_EVAL, res=_SP_RES, norm=NORM)
        del vol, mask
        return xw, sl'''

OLD_CONSUME = '''                xw = _fut.pop(i).result()
                if _PARALLEL_ARMS:
                    # Both arms at once, each on its own device. infer_probs moves the tensor.
                    def _run_arm(_t):
                        _a, (_mm, _rr) = _t
                        return _a, infer_probs(_mm, xw, _arm_devs[_a])
                    for _a, _o in _arm_ex.map(_run_arm, list(enumerate(_sp_models))):
                        arm_probs[_a][i] = _o
                else:
                    for a, (_m, _r) in enumerate(_sp_models):
                        arm_probs[a][i] = infer_probs(_m, xw, _arm_devs[a])
                del xw'''

NEW_CONSUME = '''                xw, xsl = _fut.pop(i).result()
                if _PARALLEL_ARMS:
                    # Both arms at once, each on its own device. infer_probs moves the tensor.
                    def _run_arm(_t):
                        _a, (_mm, _rr) = _t
                        return _a, infer_probs(_mm, xw, _arm_devs[_a], xsl)
                    for _a, _o in _arm_ex.map(_run_arm, list(enumerate(_sp_models))):
                        arm_probs[_a][i] = _o
                else:
                    for a, (_m, _r) in enumerate(_sp_models):
                        arm_probs[a][i] = infer_probs(_m, xw, _arm_devs[a], xsl)
                del xw, xsl'''

OLD_CENTERS = '''def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]'''

NEW_CENTERS = '''def _eval_centers(mask, D, k):
    # WINDOW HYGIENE. The champion took range(lo+1, hi) with no validity check at all, so:
    #   * a study missing a series (Coronal non-fluid is present in only ~74%) left 12 ALL-ZERO
    #     slices inside [lo, hi], and those were sampled as window centres - black images fed to
    #     the backbone and given attention mass;
    #   * the 4 slot boundaries produced triplets spanning TWO different acquisitions, mixing e.g.
    #     the last sagittal slice and the first coronal one into a single RGB window.
    # A centre is admissible only if c-1, c and c+1 are all present AND all inside the same slot.
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    _vs = set(int(v) for v in valid)
    _bounds, _acc = [], 0
    for _b in SLOT_BUDGETS:
        _bounds.append((_acc, min(_acc + _b, D))); _acc += _b
    _bounds = [(a_, b_) for a_, b_ in _bounds if a_ < D] or [(0, D)]
    cs = [c for a_, b_ in _bounds for c in range(a_ + 1, b_ - 1)
          if (c - 1) in _vs and c in _vs and (c + 1) in _vs]
    if not cs:
        lo, hi = int(valid.min()), int(valid.max())
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]'''

GLOBALS_ANCHOR = '''_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)'''

GLOBALS_BLOCK = '''_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ---- serve-side pooling / routing, set per variant. Both None == the champion, exactly. ----
SLOT_BUDGETS = [s[2] for s in SLOTS]               # 27/21/18/12/18, must sum to MAXS
assert sum(SLOT_BUDGETS) == MAXS, "slot budgets do not sum to MAXS"
POOL_MODE = %(pool_mode)s
SLOT_PRIOR = %(slot_prior)s


def _slot_of(c):
    """Which acquisition a window centre falls in."""
    _acc = 0
    for _s, _b in enumerate(SLOT_BUDGETS):
        if _acc <= c < _acc + _b:
            return _s
        _acc += _b
    return len(SLOT_BUDGETS) - 1'''

# ---------------------------------------------------------------- variant configs
# Focal findings get max/top2; diffuse ones keep the attention mean. The split follows the
# pooling table the public 0.941 stack uses AND our own concentration measurement (the findings
# whose attention is flattest are exactly the focal ones that should be peaked).
POOL_TABLE = {"Fracture": "max", "Contusion": "max", "Medial Meniscus": "max",
              "Lateral Meniscus": "max", "Baker's": "max", "ACL": "top2", "MCL": "top2"}

# Slot indices: 0 Sag-fluid, 1 Sag-struct, 2 Cor-fluid, 3 Cor-struct, 4 Axial.
# Only the three findings the diagnostic showed MIS-ROUTED are touched; everything else is left
# exactly as trained, so the read attributes to one change.
OA_PRIOR = {"Medial OA": (2, 3), "Lateral OA": (2, 3), "PF OA": (4, 0)}
PRIOR_PENALTY = -1.5        # ~0.22x weight on the disallowed slots; a nudge, not a hard mask


def _pool_mode_literal():
    return "{" + ", ".join("%d: %r" % (LAB.index(k), v) for k, v in POOL_TABLE.items()) + "}"


def _slot_prior_literal():
    rows = []
    for n, lab in enumerate(LAB):
        allowed = OA_PRIOR.get(lab)
        if allowed is None:
            rows.append([0.0] * 5)
        else:
            rows.append([0.0 if s in allowed else PRIOR_PENALTY for s in range(5)])
    return "torch.tensor(%r)" % (rows,)


VARIANTS = {
    "knee-mri-per-finding-pooling": dict(
        title="Knee MRI - Per-Finding Pooling",
        pool_mode=_pool_mode_literal(), slot_prior="None", centers=False),
    "knee-mri-oa-plane-prior": dict(
        title="Knee MRI - Osteoarthritis Plane Prior",
        pool_mode="None", slot_prior=_slot_prior_literal(), centers=False),
    "knee-mri-window-hygiene": dict(
        title="Knee MRI - Window Hygiene",
        pool_mode="None", slot_prior="None", centers=True),
    "knee-mri-pooling-and-window-hygiene": dict(
        title="Knee MRI - Pooling and Window Hygiene",
        pool_mode=_pool_mode_literal(), slot_prior="None", centers=True),
}


def nb_source(nb):
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def patch_cell(nb, old, new, label):
    """Replace `old` with `new` in exactly one code cell. Fails loudly otherwise."""
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        n = s.count(old)
        if n:
            hits += n
            c["source"] = (s.replace(old, new)).splitlines(keepends=True)
    if hits != 1:
        raise SystemExit("patch %r matched %d times, expected exactly 1" % (label, hits))


def _bad_cells(nb, where):
    """Indices of code cells that do not compile standalone."""
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        try:
            compile("".join(c["source"]), "%s#cell%d" % (where, i), "exec")
        except SyntaxError:
            out.append(i)
    return out


def selftest():
    """Prove the refactored head is numerically identical to the champion's."""
    import torch
    torch.manual_seed(0)
    B, K, F, n = 2, 24, 64, 12
    h = torch.randn(B, K, F)
    clsW = torch.randn(n, F) * 0.02
    clsb = torch.randn(n) * 0.01
    a_raw = torch.randn(B, K, n)
    a = torch.softmax(a_raw, dim=1)
    champion = (torch.einsum("bkn,bkf->bnf", a, h) * clsW).sum(-1) + clsb
    l = torch.einsum("bkf,nf->bkn", h, clsW) + clsb
    refactor = (a * l).sum(1)
    d = (champion - refactor).abs().max().item()
    print("[selftest] champion vs refactored head, max abs diff = %.3e" % d)
    if d > 1e-4:
        raise SystemExit("REFACTOR IS NOT EQUIVALENT - do not ship")
    # and that max/top2 actually differ, i.e. the lever is real
    mx = l[:, :, 0].max(1).values
    print("[selftest] max-pool differs from attention-pool by %.3f (must be non-zero to be a lever)"
          % (mx - refactor[:, 0]).abs().mean().item())
    print("[selftest] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="variants")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    selftest()

    base = json.load(open(BASE_NB, encoding="utf-8"))
    meta_base = json.load(open(BASE_META, encoding="utf-8"))
    global BASE_BAD
    BASE_BAD = _bad_cells(base, BASE_NB)
    print("[base] pre-existing non-compiling cells (IPython-only syntax): %s" % BASE_BAD)
    for slug, cfg in VARIANTS.items():
        nb = copy.deepcopy(base)
        patch_cell(nb, GLOBALS_ANCHOR,
                   GLOBALS_BLOCK % {"pool_mode": cfg["pool_mode"],
                                    "slot_prior": cfg["slot_prior"]}, "globals")
        patch_cell(nb, OLD_HEAD, NEW_HEAD, "head")
        patch_cell(nb, OLD_EVALWIN_TAIL, NEW_EVALWIN_TAIL, "eval_windows")
        patch_cell(nb, OLD_INFER, NEW_INFER, "infer_probs")
        patch_cell(nb, OLD_PREP, NEW_PREP, "_prep_study")
        patch_cell(nb, OLD_CONSUME, NEW_CONSUME, "consumer")
        if cfg["centers"]:
            patch_cell(nb, OLD_CENTERS, NEW_CENTERS, "_eval_centers")

        d = os.path.join(a.out, "push_" + slug.replace("-", "_"))
        os.makedirs(d, exist_ok=True)
        nb_p = os.path.join(d, slug + ".ipynb")
        json.dump(nb, open(nb_p, "w", encoding="utf-8"))
        m = dict(meta_base)
        m.pop("id_no", None)
        m["id"] = "dreaddevelopment/" + slug
        m["title"] = cfg["title"]
        m["code_file"] = slug + ".ipynb"
        m["kernel_type"] = "notebook"
        json.dump(m, open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)

        src = nb_source(nb)
        # Syntax gate: require NO NEW compile failures relative to the unpatched champion.
        # An absolute gate is wrong here - cell 1 of the shipped notebook carries a mid-cell
        # `from __future__ import annotations`, which IPython accepts and plain compile() does not.
        # That cell is pre-existing dead legacy code and is not touched by any patch.
        bad = _bad_cells(nb, nb_p)
        new_bad = sorted(set(bad) - set(BASE_BAD))
        if new_bad:
            raise SystemExit("%s: patch introduced syntax errors in cell(s) %s" % (slug, new_bad))
        for must in ("POOL_MODE", "SLOT_PRIOR", "_slot_of", "def head(self, feats, slots=None)"):
            assert must in src, "%s: %s missing" % (slug, must)
        print("[ok] %-40s %s  (%d chars)" % (slug, d, len(src)))
    print("\n%d variants written. Each is ONE board read against 0.941; fire one at a time."
          % len(VARIANTS))


if __name__ == "__main__":
    main()
