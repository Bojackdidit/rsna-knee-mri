#!/usr/bin/env python3
"""Build the 'more augmented views' variant from the standing 0.940 champion.

Companion to the denser-window candidate and independent of it. Densifying windows is bounded by
MEMORY (the slice cache grows with the test set), so it can only reach ~1.8x before it must stop.
Averaging more augmented views of the SAME windows costs TIME and no memory at all, so it can use
the budget densification cannot touch. Our measured run is 3 h 32 m against an 8 h TIME_BUDGET.

Exactly one behaviour changes: a jittered member averages N_VIEWS views per window instead of 2.
The scheduler's cost model is updated in the same edit and in the same units, so it keeps
self-limiting: if the extra views do not fit, `jit` goes False for that member and it falls back to
a single view rather than overrunning. That is the safe failure direction.
"""
import json, sys, os, difflib

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_views"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SLUG = "knee-mri-more-augmented-views"
TITLE = "Knee MRI - More Augmented Views"

EDITS = [
    # 1. the constant, introduced immediately above the function that uses it
    ("@torch.no_grad()\ndef predict_member(model, cache, mask, idx, dev, img_size,",
     f"# Views averaged per window for a jittered member (champion behaviour is 2: original + 1).\n"
     f"# Raised to spend inference budget that the memory-bound window count cannot use.\n"
     f"N_VIEWS = {N}\n"
     f"log(f'TTA views per jittered window: {{N_VIEWS}}')\n\n"
     "@torch.no_grad()\ndef predict_member(model, cache, mask, idx, dev, img_size,"),
    # 2. produce N_VIEWS - 1 jittered views instead of one
    ("            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])\n",
     "            views = [rows] + ([augment(rows, generator=jitter_gen)\n"
     "                               for _ in range(N_VIEWS - 1)] if jitter else [])\n"),
    # 3. the measured per-pass cost must count the same number of passes
    ("    passes = len(starts) * (2 if jitter else 1)\n",
     "    passes = len(starts) * (N_VIEWS if jitter else 1)\n"),
    # 4-5. the scheduler prices jitter at the real number of views
    ("                    jit = est['fixed'] + 2 * len(starts_full) * est['win'] <= room * 0.6\n",
     "                    jit = est['fixed'] + N_VIEWS * len(starts_full) * est['win'] <= room * 0.6\n"),
    ("                    per_win = est['win'] * (2 if jit else 1)\n",
     "                    per_win = est['win'] * (N_VIEWS if jit else 1)\n"),
]

nb = json.load(open(SRC, encoding="utf-8"))
counts = [0] * len(EDITS)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for k, (old, new) in enumerate(EDITS):
        n = j.count(old)
        if n:
            j = j.replace(old, new, 1)
            counts[k] += 1
    c["source"] = j.splitlines(keepends=True)
for k, (old, _) in enumerate(EDITS):
    assert counts[k] == 1, f"edit {k} matched {counts[k]} times (expected 1): {old[:60]!r}"

import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all patched cells parse")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)

old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
print("\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1)))
print(f"\nall {len(EDITS)} anchors matched exactly once")
