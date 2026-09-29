#!/usr/bin/env python3
"""Two-group blend weight: our half high everywhere EXCEPT the two findings the partner wins.

Measured on the 58-study gold gate (our confdist arm vs partner_oof.csv), per finding:
    ours   partner
    MCL              0.9955  0.6077   <- our arm is not close to being beaten
    ACL              0.9743  0.9007
    Medial Meniscus  0.9675  0.8089
    Baker's          0.9728  0.8859
    Effusion         0.9540  0.8522
    ... our arm wins 10 of 12 ...
    Lateral OA       0.8201  0.8627   <- partner wins
    PF OA            0.8417  0.8842   <- partner wins

Free per-finding optimisation scores 0.9356 against 0.9099 for the live uniform 0.55. I am NOT
shipping those fitted weights: twelve weights fitted on 58 studies is overfitting, and at n=58 the
AUC standard error is ~0.05-0.08, so most individual gaps are inside noise.

What IS robust is the SHAPE - a 0.39 AUC gap on MCL is not noise, and the reversal is confined to
the two osteoarthritis findings. So this is a TWO-parameter change, not twelve: one high weight for
the ten findings our arm wins, one low weight for Lateral OA and PF OA. That keeps the degrees of
freedom low enough to survive the gate's noise while capturing the effect that is clearly real.
"""
import ast, difflib, json, os, sys

HI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_armw2g"
SLUG = "knee-mri-our-half-two-group"
TITLE = "Knee MRI - Our Half Two Group"
LOW_FINDINGS = ("Lateral OA", "PF OA")

OLD_BASE = """    _coatnet_weight = {
        label: 0.55
        for label in _blend_labels
    }"""
NEW_BASE = f"""    _coatnet_weight = {{
        label: ({LO} if label in ('Lateral OA', 'PF OA') else {HI})
        for label in _blend_labels
    }}"""
# The V18 bumps assume a 0.55 base; at this base they would REDUCE those findings, so they are
# folded into the high group instead of left to invert.
OLD_V18 = """            {
                'Medial Meniscus': 0.57,
                'Lateral Meniscus': 0.59,
                'Fracture': 0.59,
            }"""
NEW_V18 = f"""            {{
                'Medial Meniscus': {min(HI + 0.02, 1.0)},
                'Lateral Meniscus': {min(HI + 0.04, 1.0)},
                'Fracture': {min(HI + 0.04, 1.0)},
            }}"""

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"base": 0, "v18": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD_BASE in j: j = j.replace(OLD_BASE, NEW_BASE, 1); hits["base"] += 1
    if OLD_V18 in j:  j = j.replace(OLD_V18, NEW_V18, 1);  hits["v18"] += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == {"base": 1, "v18": 1}, hits
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True, "must stay private"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print(f"HI={HI} LO={LO}: parses OK, private OK, {len(d)} changed lines")
for l in d: print("   ", l.strip())
