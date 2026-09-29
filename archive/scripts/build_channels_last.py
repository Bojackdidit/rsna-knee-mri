#!/usr/bin/env python3
"""ConvNeXt-Base 384 pair in channels_last. v2 - patched at the right tensor.

v1 put the format change on the infer_probs input, which is RANK 5 (B, K, 3, H, W) - a bag of K
windows per study. channels_last requires rank 4, so it threw on every study, the caller's FALLBACK
skipped the model entirely (3 studies in 11s against 50s for a real run), and the output changed
because nothing was inferred. Caught by comparing the commit run against the previous one.

The rank-4 tensor is inside RaptorClassifier.encode, where the window bag is flattened before the
backbone. That is where the format belongs.

Why bother at all: three board reads show every step DOWN in model capacity costs ~4.4x more accuracy
than the time it buys (0.001 AUC = 72 s). Cheapening cannot give "faster AND better". channels_last
changes memory layout, not capacity - NHWC lets cuDNN hit Turing fp16 tensor-core convs without
transposing per layer, and ConvNeXt is pure conv, the case that gains most.
"""
import io, json, os, sys, ast

SRC  = "variants/push_knee_mri_convnext_base_384/knee-mri-convnext-base-384.ipynb"
OUT  = "variants/push_cnb384_cl"
SLUG = "knee-mri-convnext-base-384-fast"
TITLE = "Knee MRI - ConvNeXt-Base 384 Fast"

OLD_ENC = ("    def encode(self, x):\n"
           "        B, K = x.shape[:2]\n"
           "        f = self.backbone(x.flatten(0, 1))\n"
           "        return f.view(B, K, -1)\n")
NEW_ENC = ("    def encode(self, x):\n"
           "        B, K = x.shape[:2]\n"
           "        xf = x.flatten(0, 1)\n"
           "        if xf.is_cuda and xf.dim() == 4:\n"
           "            # NHWC here, where the tensor is finally rank 4. Same arithmetic; cuDNN just\n"
           "            # stops transposing every conv. Rank 5 (B,K,3,H,W) cannot carry this format.\n"
           "            xf = xf.contiguous(memory_format=torch.channels_last)\n"
           "        f = self.backbone(xf)\n"
           "        return f.view(B, K, -1)\n")

OLD_MODEL = "    model.eval().to(device)\n"
NEW_MODEL = ("    model.eval().to(device)\n"
             "    if str(device).startswith('cuda'):\n"
             "        model = model.to(memory_format=torch.channels_last)\n")

nb = json.load(io.open(SRC, encoding="utf-8"))
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
for n, cond in [("cl at encode",   "xf.contiguous(memory_format=torch.channels_last)" in body),
                ("cl on model",    "model.to(memory_format=torch.channels_last)" in body),
                ("rank-4 guard",   "xf.dim() == 4" in body),
                ("no rank-5 patch","x.contiguous(memory_format=torch.channels_last)" not in body),
                ("both cnb arms",  body.count("raptor_ft_raptor384_cnb") == 2),
                ("res 384",        body.count('"res": 384') == 2),
                ("fastdcm intact", "_fd_read_one" in body),
                ("autocast intact",'torch.autocast("cuda", dtype=torch.float16)' in body),
                ("K_EVAL 48",      "K_EVAL = 48" in body)]:
    print("  %-18s %s" % (n, "OK" if cond else "FAIL"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(io.open("variants/push_knee_mri_convnext_base_384/kernel-metadata.json", encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
meta.pop("id_no", None)
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("built", OUT)
