#!/usr/bin/env python3
"""ONE candidate built to clear BOTH bars: >= 0.940 and <= 1 hour of scoring.

Cost model measured from our own run: ~588 model passes/study over 1,322 studies took 3 h 32 m,
i.e. ~62 passes/sec, so a 1-hour budget is roughly 170 passes/study - AND that ignores DICOM decode,
which is a large FIXED cost ( measured the slice-ordering pass alone at 12 of 32 minutes).
Cutting windows alone only reaches ~1.5 h, so this cuts passes AND decode AND caps the clock.

  1. TIME_BUDGET 8 h -> 55 min. The kernel already has an affordability scheduler that SURRENDERS
     members rather than overrunning, so this makes the partner half structurally unable to exceed
     the bar instead of merely intended to.
  2. Arm loop time guard. The arm loop had NO budget check at all - it would happily run past any
     cap. It now drops to fewer windows when it is running late, and stops adding arms it cannot
     afford.
  3. Partner: 10 windows -> centre 3, and jitter OFF (2 views -> 1). 20x10x2=400 -> 20x3x1=60.
  4. Arm: K_EVAL 94 -> 32, drawn per-sequence from the middle. 94x2=188 -> 32x2=64.
  5. Decode halved: SLOTS slice counts 27/21/18/12/18 (96) -> 14/11/9/6/9 (49). Same SPAN, fewer
     samples across it. The span is deliberately NOT narrowed - the champion's own comment warns the
     collateral ligaments and lateral meniscus live in the peripheral slices, and narrowing would
     also break the geometry the weights were fitted on.

Total ~124 passes/study against 588, plus about half the decode.
"""
import ast, difflib, json, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_sub1h"
SLUG = "knee-mri-under-one-hour"
TITLE = "Knee MRI - Under One Hour"

EDITS = [
    # 1. hard clock
    ("TIME_BUDGET = 8.0 * 3600",
     "TIME_BUDGET = 55.0 * 60   # hard bar: scoring must finish inside one hour"),
    # 5. decode half as many slices per sequence, same span
    ('SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),\n         ("Coronal", 0, 12), ("Axial", -1, 18)]',
     '# Halved slice counts: same span across each series, sampled half as densely. Decode is a large\n'
     '# fixed cost, so this is the only lever that cuts the part window trimming cannot touch.\n'
     'SLOTS = [("Sagittal", 1, 14), ("Sagittal", 0, 11), ("Coronal", 1, 9),\n         ("Coronal", 0, 6), ("Axial", -1, 9)]'),
    # 4. arm windows
    ("K_EVAL = 94", "K_EVAL = 32"),
    # 3. partner windows -> centre 3
    ("        starts_full = window_starts(Cte.shape[2], GROUP)\n",
     "        starts_full = window_starts(Cte.shape[2], GROUP)\n"
     "        if len(starts_full) > 3:\n"
     "            _mid = (len(starts_full) - 3) // 2\n"
     "            starts_full = starts_full[_mid:_mid + 3]\n"
     "        log(f'SUB1H: partner windows -> {len(starts_full)} centre position(s) {starts_full}')\n"),
    # 3b. partner jitter off (halves partner passes)
    ("                    jit = est['fixed'] + 2 * len(starts_full) * est['win'] <= room * 0.6\n",
     "                    jit = False   # SUB1H: one view per window, not two\n"),
    # 2. arm loop time guard
    ("        for i, sid in enumerate(test_ids):\n            try:\n                vol, mask = build_study(sid, SER, tsdir, reader)\n                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)",
     "        _k_arm = K_EVAL\n"
     "        for i, sid in enumerate(test_ids):\n"
     "            # SUB1H guard: the arm loop previously had no budget check at all. If we are\n"
     "            # tracking late, halve the windows rather than blow the one-hour bar.\n"
     "            if i and i % 100 == 0:\n"
     "                _spent = time.time() - T0\n"
     "                _projected = _spent / max(i, 1) * N * (A - a) + _spent\n"
     "                if _projected > TIME_BUDGET and _k_arm > 8:\n"
     "                    _k_arm = max(8, _k_arm // 2)\n"
     "                    print(f'  [arm {a}] running late ({_spent/60:.0f} min); windows -> {_k_arm}', flush=True)\n"
     "            try:\n"
     "                vol, mask = build_study(sid, SER, tsdir, reader)\n"
     "                xw = eval_windows(vol, mask, k=_k_arm, res=res, norm=NORM)"),
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
bad = [i for i, n in enumerate(counts) if n != 1]
assert not bad, f"edits {bad} did not match exactly once: {counts}"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print("all %d edits matched exactly once; all cells parse; private OK" % len(EDITS))
print("passes/study: 20x3x1 partner + 32x2 arm = %d  (was 588, %.1fx less)" % (20*3*1 + 32*2, 588/(20*3+64)))
print("decoded slices/study: 49 (was 96)")
print("%d changed lines -> %s" % (len(d), OUT_DIR))
