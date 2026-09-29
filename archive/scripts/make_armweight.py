#!/usr/bin/env python3
"""Sweep the weight our CoAtNet half carries against the partner (transformer) half.

Why this is the strongest untested lever we have:
  - Board history only ever read TWO points: 0.50 -> 0.938 and 0.55 -> 0.939/0.940. Rising.
  - Measured offline on the 58-study gold gate (partner_oof.csv vs our confdist arm):
        partner half alone   0.8375
        our arm alone        0.9266
        best mixed blend     0.9212  (at ours 0.70 / partner 0.30)
    i.e. every mixture is WORSE than our arm alone, and the curve rises monotonically toward us.
  - Board trend and gold gate agree on the DIRECTION, which is rare here; the gold gate alone has
    mispredicted the board repeatedly, so direction-agreement is the thing worth acting on.
  - w=1.00 makes the partner half contribute nothing, i.e. a SINGLE-MODEL submission. A top
    competitor reportedly scores 0.944 from one model, so this reads our single model against that
    claim directly instead of guessing at it.

Single variable: the CoAtNet share. The V18 per-finding calibration bumps three findings above the
base (Medial Meniscus +0.02, Lateral Meniscus +0.04, Fracture +0.04); those offsets are preserved
relative to the new base and clipped to 1.0, so the calibration's intent survives the sweep instead
of silently inverting at high weights.
"""
import ast, difflib, json, os, sys

W = float(sys.argv[1])
SRC = "champ/knee-mri-blend-even-full.ipynb"
tag = "%03d" % round(W * 100)
OUT_DIR = f"variants/push_armw{tag}"
SLUG = f"knee-mri-our-half-weight-{tag}"
TITLE = f"Knee MRI - Our Half Weight {tag}"

OLD_BASE = """    _coatnet_weight = {
        label: 0.55
        for label in _blend_labels
    }"""
NEW_BASE = f"""    _coatnet_weight = {{
        label: {W}
        for label in _blend_labels
    }}"""

OLD_V18 = """            {
                'Medial Meniscus': 0.57,
                'Lateral Meniscus': 0.59,
                'Fracture': 0.59,
            }"""
NEW_V18 = f"""            {{
                'Medial Meniscus': {min(W + 0.02, 1.0)},
                'Lateral Meniscus': {min(W + 0.04, 1.0)},
                'Fracture': {min(W + 0.04, 1.0)},
            }}"""

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"base": 0, "v18": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD_BASE in j:
        j = j.replace(OLD_BASE, NEW_BASE, 1); hits["base"] += 1
    if OLD_V18 in j:
        j = j.replace(OLD_V18, NEW_V18, 1); hits["v18"] += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == {"base": 1, "v18": 1}, f"anchors matched {hits} (expected 1 each)"

for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
assert meta.get("is_private") is True, "kernel must stay private (no publishing before first place)"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)

old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
print(f"w={W}: parses OK, private OK, {len(d)} changed lines -> {OUT_DIR}")
for l in d: print("   ", l.strip())
