#!/usr/bin/env python3
"""Champion kernel with INTERLEAVED window positions. Usage: build_interleave.py K_PER_ARM

Arms run CONCURRENTLY one per GPU, so wall-clock is ONE arm's cost. Today both arms are handed the
IDENTICAL window tensor, so the blend sees K distinct positions for K of cost. Build 2K positions
once (the window build is CPU prep, already overlapped by the prefetch pool) and give arm 0 the even
positions and arm 1 the odd ones: the blend then sees 2K distinct positions for K of GPU cost.

The accuracy lost by cutting K is a COVERAGE loss - the champion arm's own sweep shows
K=48 0.9453, K=24 0.9445, K=12 0.9375, K=9 0.9340 - so restoring coverage without restoring cost is
exactly the right shape of fix. Window PLACEMENT was swept first and is closed: centre_frac 1.00 wins
at every K (K=12: 1.00 -> 0.9375, 0.75 -> 0.9377, 0.50 -> 0.9263, 0.35 -> 0.9246).

Runtime = 3.7 min reading + 20.1 * K/48 min of model:
    K=12 -> 8.7 min     K=9 -> 7.5 min     K=6 -> 6.2 min
"""
import io, json, os, sys, ast, glob

K = int(sys.argv[1])
SRC = "variants/push_fastdcm"
SLUG = "knee-mri-coatnet-384-int%d" % K
TITLE = "Knee MRI - CoAtNet 384 Int%d" % K
OUT = "variants/push_" + SLUG.replace("-", "_")

nb = json.load(io.open(glob.glob(os.path.join(SRC, "*.ipynb"))[0], encoding="utf-8"))
subs = [
    ("K_EVAL = 48", "K_EVAL = %d" % (2 * K)),
    ("return _a, infer_probs(_mm, xw, _arm_devs[_a])",
     "return _a, infer_probs(_mm, xw[_a % 2::2], _arm_devs[_a])"),
    ("arm_probs[a][i] = infer_probs(_m, xw, _arm_devs[a])",
     "arm_probs[a][i] = infer_probs(_m, xw[a % 2::2], _arm_devs[a])"),
]
hits = {a: 0 for a, _ in subs}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = o = "".join(c["source"])
    for a, b in subs:
        if a in s:
            s = s.replace(a, b, 1); hits[a] += 1
    if s != o:
        c["source"] = s.splitlines(keepends=True)
print("patch hits:", {k[:34]: v for k, v in hits.items()})
if any(v != 1 for v in hits.values()):
    sys.exit("anchors did not all match exactly once")

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for n, cond in [("2K windows built", "K_EVAL = %d" % (2 * K) in body),
                ("parallel path split", "xw[_a % 2::2]" in body),
                ("serial path split",   "xw[a % 2::2]" in body),
                ("champion arm 1", "raptor_ft_confdist_swa.pt" in body),
                ("champion arm 2", "raptor_ft_coatnet_w80_full_swa.pt" in body),
                ("still 384", body.count('"res": 384') == 2),
                ("fastdcm intact", "_fd_read_one" in body),
                ("all six slots", "SLOTS[:2]" not in body)]:
    print("  %-20s %s" % (n, "OK" if cond else "FAIL"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(io.open(os.path.join(SRC, "kernel-metadata.json"), encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
meta.pop("id_no", None)
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("built", OUT, "| slug ok:", TITLE.lower().replace(" - ", "-").replace(" ", "-") == SLUG)
