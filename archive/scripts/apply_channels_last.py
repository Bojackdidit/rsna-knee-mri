#!/usr/bin/env python3
"""Apply channels_last to an already-built arm kernel. Usage: apply_channels_last.py SRC_DIR SLUG TITLE

Patches inside RaptorClassifier.encode, where the window bag is flattened to rank 4. Patching the
infer_probs input instead does NOT work - it is rank 5 (B,K,3,H,W), channels_last throws, and the
caller's FALLBACK silently skips the model entirely. That failure looked like a successful run.

Verified numerically exact on the ConvNeXt-Base pair: byte-identical submission to the unpatched run,
zero fallbacks. So this is a pure speed change and the score is unaffected.
"""
import io, json, os, sys, ast, glob

SRC_DIR, SLUG, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = "variants/push_" + SLUG.replace("-", "_")
src_nb = glob.glob(os.path.join(SRC_DIR, "*.ipynb"))[0]

OLD_ENC = ("    def encode(self, x):\n"
           "        B, K = x.shape[:2]\n"
           "        f = self.backbone(x.flatten(0, 1))\n"
           "        return f.view(B, K, -1)\n")
NEW_ENC = ("    def encode(self, x):\n"
           "        B, K = x.shape[:2]\n"
           "        xf = x.flatten(0, 1)\n"
           "        if xf.is_cuda and xf.dim() == 4:\n"
           "            xf = xf.contiguous(memory_format=torch.channels_last)\n"
           "        f = self.backbone(xf)\n"
           "        return f.view(B, K, -1)\n")
OLD_MODEL = "    model.eval().to(device)\n"
NEW_MODEL = ("    model.eval().to(device)\n"
             "    if str(device).startswith('cuda'):\n"
             "        model = model.to(memory_format=torch.channels_last)\n")

nb = json.load(io.open(src_nb, encoding="utf-8"))
h = {"enc": 0, "model": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = o = "".join(c["source"])
    if OLD_ENC in s:
        s = s.replace(OLD_ENC, NEW_ENC, 1); h["enc"] += 1
    if OLD_MODEL in s:
        s = s.replace(OLD_MODEL, NEW_MODEL, 1); h["model"] += 1
    if s != o:
        c["source"] = s.splitlines(keepends=True)
print("patch hits:", h)
if any(v != 1 for v in h.values()):
    sys.exit("anchors did not match exactly once")
body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for n, cond in [("cl at encode", "xf.contiguous(memory_format=torch.channels_last)" in body),
                ("cl on model",  "model.to(memory_format=torch.channels_last)" in body),
                ("rank-4 guard", "xf.dim() == 4" in body),
                ("no rank-5 patch", "x.contiguous(memory_format=torch.channels_last)" not in body),
                ("fastdcm intact", "_fd_read_one" in body),
                ("autocast intact", 'torch.autocast("cuda", dtype=torch.float16)' in body),
                ("K_EVAL 48", "K_EVAL = 48" in body)]:
    print("  %-18s %s" % (n, "OK" if cond else "FAIL"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(io.open(os.path.join(SRC_DIR, "kernel-metadata.json"), encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
meta.pop("id_no", None)
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("built", OUT, "| slug ok:", TITLE.lower().replace(" - ", "-").replace(" ", "-") == SLUG)
