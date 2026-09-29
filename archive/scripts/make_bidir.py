#!/usr/bin/env python3
"""Bidirectional slice-axis TTA: also read each 3-slice window in reverse order.

Every window we feed the model is 3 physically-consecutive slices stacked as RGB. The model has
therefore only ever seen them in one direction. The public fork of the 0.936 stack carries a
forward/reversed combination at weights 0.7857 / 0.2143 that our arm does not have.

⚠ Honest status: the fork ships NO ablation for this, so it is an UNTESTED claim, not a measured
gain. It is queued as a board read precisely because it cannot be settled offline - our gold-58 gate
has now mispredicted the board seven times, so an offline screen here would be worthless.

Why it is a reasonable read anyway: it is a genuinely new TTA axis (we already average over window
positions and a jittered view, but never over slice direction), it costs time and no memory, and our
measured run finishes in 3 h 32 m of an 8 h budget. The reversed view is weighted well below the
forward one, so the failure direction is mild.

The forward views keep their combined 0.7857 share (split evenly when jitter adds a second forward
view), and the scheduler's cost model counts the extra pass so the affordability gate stays honest.
"""
import ast, difflib, json, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_bidir"
SLUG = "knee-mri-bidirectional-slice-tta"
TITLE = "Knee MRI - Bidirectional Slice TTA"
FWD, REV = 0.7857, 0.2143

EDITS = [
    # 1. constants, introduced above the function that uses them
    ("@torch.no_grad()\ndef predict_member(model, cache, mask, idx, dev, img_size,",
     f"# Bidirectional slice-axis TTA: each 3-slice window is also read reversed, at the public\n"
     f"# fork's weights. Forward views share {FWD}; the reversed view takes {REV}.\n"
     f"BIDIR_FWD_W = {FWD}\nBIDIR_REV_W = {REV}\n"
     f"log(f'bidirectional slice TTA: forward {{BIDIR_FWD_W}} / reversed {{BIDIR_REV_W}}')\n\n"
     "@torch.no_grad()\ndef predict_member(model, cache, mask, idx, dev, img_size,"),
    # 2. add the reversed view and carry per-view weights
    ("            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])\n",
     "            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])\n"
     "            # slice axis is dim 2 (cache[sel, :, st:st+group]), so flipping it reverses the\n"
     "            # 3-slice order the model reads.\n"
     "            _vw = [BIDIR_FWD_W / len(views)] * len(views) + [BIDIR_REV_W]\n"
     "            views = views + [torch.flip(rows, dims=(2,))]\n"),
    # 3-4. weighted combination instead of a plain mean
    ("            win_logits.append(torch.stack(view_logits).mean(0))\n"
     "            win_probs.append(torch.stack(view_probs).mean(0))\n",
     "            _w = torch.tensor(_vw, device=view_probs[0].device,\n"
     "                              dtype=view_probs[0].dtype).view(-1, *([1] * view_probs[0].dim()))\n"
     "            win_logits.append((torch.stack(view_logits) * _w).sum(0))\n"
     "            win_probs.append((torch.stack(view_probs) * _w).sum(0))\n"),
    # 5. the measured per-pass cost must count the extra reversed pass
    ("    passes = len(starts) * (2 if jitter else 1)\n",
     "    passes = len(starts) * ((2 if jitter else 1) + 1)\n"),
    # 6-7. the scheduler prices the extra pass too
    ("                    jit = est['fixed'] + 2 * len(starts_full) * est['win'] <= room * 0.6\n",
     "                    jit = est['fixed'] + 3 * len(starts_full) * est['win'] <= room * 0.6\n"),
    ("                    per_win = est['win'] * (2 if jit else 1)\n",
     "                    per_win = est['win'] * ((2 if jit else 1) + 1)\n"),
]

nb = json.load(open(SRC, encoding="utf-8"))
counts = [0] * len(EDITS)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for k, (old, new) in enumerate(EDITS):
        if old in j:
            j = j.replace(old, new, 1); counts[k] += 1
    c["source"] = j.splitlines(keepends=True)
bad = [k for k, n in enumerate(counts) if n != 1]
assert not bad, f"edits {bad} did not match exactly once: {counts}"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse; every anchor matched exactly once")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True, "must stay private (no publishing before first place)"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print(f"private OK, {len(d)} changed lines -> {OUT_DIR}")
