#!/usr/bin/env python3
"""Make the kernel plan eval windows WITHIN a series, never across two. Usage:
   patch_slotwindows.py SRC_NOTEBOOK OUT_NOTEBOOK

The kernel stacks five different MRI series end to end into one 96-slice volume (27 sag-fluid,
21 sag-nonfluid, 18 cor-fluid, 12 cor-nonfluid, 18 axial) and then plans window centres by
linspacing between the first and last valid slice of the WHOLE volume. A window is three
physically adjacent slices shown to the model as RGB, so any centre within one slice of a slot
boundary builds its triple out of two different series - a picture of nothing.

Measured on the shipped geometry: 5 of 48 windows (10.4%) are such windows, and it gets worse as K
falls - 16.7% at K=12, which is exactly the regime the efficiency track pushes us into. That is a
plausible part of why the champion arms dropped 0.941 -> 0.931 when K was cut to 12.

The identical bug was found and fixed on the training side (finetune_box `_centers`); this is the
serving half of the same defect. No retraining, no extra runtime: the same number of windows is
produced, they are just all real.
"""
import json, sys, io, re

SRC, OUT = sys.argv[1], sys.argv[2]

NEW = '''def _eval_centers(mask, D, k):
    # Plan centres WITHIN each slot. The volume concatenates five separate MRI series, so a centre
    # within one slice of a boundary builds its RGB triple from two different series. Allocating k
    # across slots in proportion to how many legal centres each holds keeps the same window count
    # and the same cost, and every window is now three slices of one series.
    valid = set(int(v) for v in np.where(mask > 0)[0])
    edges, acc = [0], 0
    for _p, _f, _n in SLOTS:
        acc += _n; edges.append(min(acc, D))
    per = []
    for s in range(len(edges) - 1):
        a, b = int(edges[s]), int(edges[s + 1])
        cand = [c for c in range(a + 1, b - 1)
                if (c - 1) in valid and c in valid and (c + 1) in valid]
        per.append(cand)
    if not any(per):
        # No slot holds three consecutive valid slices. The old fallback linspaced across the WHOLE
        # volume, which re-introduced exactly the cross-series windows this function exists to stop -
        # a 30,000-mask fuzz found it on 2,565 degenerate masks. Stay inside ONE slot instead: pick
        # the slot with the most valid slices and centre the windows in it. The triple may include an
        # invalid slice of that series, which is merely uninformative; it can never span two series,
        # which is actively wrong.
        best_s, best_n = 0, -1
        for s in range(len(edges) - 1):
            a, b = int(edges[s]), int(edges[s + 1])
            n = sum(1 for c in range(a, b) if c in valid)
            if n > best_n:
                best_s, best_n = s, n
        a, b = int(edges[best_s]), int(edges[best_s + 1])
        if b - a >= 3:
            cs = list(range(a + 1, b - 1))
        else:
            cs = [max(a, min((a + b) // 2, max(b - 2, a)))]
        cs = [max(1, min(int(c), D - 2)) for c in cs] or [max(1, min(D // 2, D - 2))]
        idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
        return [cs[i] for i in idx]
    tot = sum(len(c) for c in per)
    want = [int(round(k * len(c) / tot)) for c in per]
    # repair rounding so the counts sum to exactly k, only ever giving to slots that have candidates
    while sum(want) != k:
        d = 1 if sum(want) < k else -1
        order = sorted(range(len(per)), key=lambda i: -len(per[i]))
        for i in order:
            if not len(per[i]):
                continue
            if d < 0 and want[i] <= 0:
                continue
            want[i] += d
            break
        else:
            break
    out = []
    for c, w in zip(per, want):
        if w <= 0 or not c:
            continue
        idx = np.linspace(0, len(c) - 1, w).round().astype(int)
        out.append([c[i] for i in idx])
    # interleave the slots so that the kernel's even/odd arm split still gives each arm a share of
    # every series rather than handing one arm the sagittal slices and the other the axial ones
    cs = []
    for j in range(max((len(o) for o in out), default=0)):
        for o in out:
            if j < len(o):
                cs.append(o[j])
    return cs[:k] if len(cs) >= k else (cs + [cs[-1]] * (k - len(cs)) if cs else [max(1, D // 2)])'''

nb = json.load(io.open(SRC, encoding="utf-8"))
hit = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    m = re.search(r"def _eval_centers\(mask, D, k\):.*?(?=\n\ndef |\n\n# =)", src, re.S)
    if not m:
        continue
    src = src[:m.start()] + NEW + src[m.end():]
    cell["source"] = [l + "\n" for l in src.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    hit += 1
if hit != 1:
    sys.exit("expected exactly one _eval_centers, found %d" % hit)
json.dump(nb, io.open(OUT, "w", encoding="utf-8"), indent=1)
print("patched %d cell -> %s" % (hit, OUT))
