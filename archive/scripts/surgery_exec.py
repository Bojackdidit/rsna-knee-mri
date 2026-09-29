#!/usr/bin/env python3
"""Execution-only surgery on a built kernel. Usage: surgery_exec.py DIR

Score-frozen by construction: none of this changes which slices are read, which windows are chosen,
or the order anything is consumed in. It changes only WHERE work happens and HOW MANY threads do it.

Measured justification (all our own board reads):
  - EffNetB0+B1@288 is ~1/20 the compute of CoAtNet-384 and bought 2 minutes, so runtime is the
    per-study pipeline, not the backbone.
  - Raising study-prefetch 2 -> 6 bought 35 s and the kernel reports `cpus 4`: the four cores were
    already saturated. So the fix is to do LESS CPU work per study, not to add more threads to it.

Three changes:
 1. eval_windows built a (K,3,res,res) **float32** array on CPU - 48x3x384x384 = 84.9 MB per study -
    doing a per-window 336->384 bilinear AND the ImageNet normalise on CPU. It now emits uint8 at the
    cache's native resolution: 48x3x336x336 = 16.3 MB, 5.2x less to build and copy. The resize and
    normalise move to the GPU, which is idle while the CPU is the bottleneck.
    (Our cache is 336 px, so every "384" window was an upsample of 336 anyway - no detail is lost.)
 2. The host->device copy becomes pinned + non_blocking instead of pageable + synchronous.
 3. Reader threads 16 -> 48. a published result measured header-thread saturation at 64 on this dataset
    (discussion/736678); the wall is per-file open latency on gcsfuse, not bytes.
"""
import ast, glob, io, json, os, sys

D = sys.argv[1]
nbp = glob.glob(os.path.join(D, "*.ipynb"))[0]
nb = json.load(io.open(nbp, encoding="utf-8"))

OLD_EW = '''def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x'''

NEW_EW = '''def eval_windows(vol, mask, k, res, norm=NORM):
    # Emit uint8 at the volume's NATIVE resolution and let the GPU do the float conversion, the
    # resize and the normalisation. The centres and the slices are identical to before; only the
    # dtype and the device the arithmetic happens on change.
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    h, w = vol.shape[-2], vol.shape[-1]
    wins = np.empty((len(cs), 3, h, w), np.uint8)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        wins[j, 0] = vol[c - 1]
        wins[j, 1] = vol[c]
        wins[j, 2] = vol[c + 1]
    x = torch.from_numpy(wins)
    try:
        x = x.pin_memory()
    except Exception:
        pass
    return x'''

OLD_IP = '''def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)'''

NEW_IP = '''def _to_model_input(xwins, device, res, norm=NORM):
    """uint8 CPU windows -> normalised float windows on `device`, resized if needed."""
    x = xwins.to(device, non_blocking=True)
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    if x.shape[-1] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    if norm == "imagenet":
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
    return x


@torch.no_grad()
def infer_probs(model, xwins, device, res=None):
    x = _to_model_input(xwins, device, res if res is not None else xwins.shape[-1]).unsqueeze(0)'''

body_before = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
subs = [(OLD_EW, NEW_EW), (OLD_IP, NEW_IP),
        ('os.environ.get("ARM_RD_THREADS", "16")', 'os.environ.get("ARM_RD_THREADS", "48")')]
hits = [body_before.count(a) for a, _ in subs]
print("  anchor hits:", hits)

for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    o = s
    for a, b in subs:
        s = s.replace(a, b)
    if s != o:
        c["source"] = [l + chr(10) for l in s.split(chr(10))]

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
# the call sites must pass the model's resolution now that the tensor arrives at native size
body_calls = body.count("infer_probs(")
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    s = s.replace("infer_probs(_mm, xw[_a % 2::2], _arm_devs[_a])",
                  "infer_probs(_mm, xw[_a % 2::2], _arm_devs[_a], _rr)")
    s = s.replace("infer_probs(_m, xw[a % 2::2], _arm_devs[a])",
                  "infer_probs(_m, xw[a % 2::2], _arm_devs[a], _r)")
    s = s.replace("infer_probs(_mm, xw, _arm_devs[_a])", "infer_probs(_mm, xw, _arm_devs[_a], _rr)")
    s = s.replace("infer_probs(_m, xw, _arm_devs[a])", "infer_probs(_m, xw, _arm_devs[a], _r)")
    c["source"] = [l + chr(10) for l in s.split(chr(10))]

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
checks = [
    ("eval_windows emits uint8", "np.uint8)" in body and "_to_model_input" in body),
    ("no CPU interpolate in eval_windows", "wins[j] = t.numpy()" not in body),
    ("pinned + non_blocking upload", "pin_memory()" in body and "non_blocking=True" in body),
    ("reader threads 48", '"ARM_RD_THREADS", "48"' in body),
    ("infer_probs call sites carry res", body.count("_arm_devs[_a], _rr") + body.count("_arm_devs[a], _r") >= 1),
    ("call count unchanged", body.count("infer_probs(") == body_calls),
    ("every cell parses", True),
]
for n, ok in checks:
    print("  %-34s %s" % (n, "OK" if ok else "FAIL"))
assert all(ok for _, ok in checks), "surgery failed its own checks"
json.dump(nb, io.open(nbp, "w", encoding="utf-8"))
print("  wrote", nbp)
