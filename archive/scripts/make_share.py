#!/usr/bin/env python3
"""Scale the legacy family's share of the blend, to bracket it in one day instead of one a week.

The family is given weight on four findings (Lateral Meniscus 15, Lateral OA 15, Contusion 5,
Medial OA 2.5) against the 20 main members' 1.0 each — so at full share it takes ~75% of the first
two. That is the public recipe's tuning, not ours, and it is the single number most likely to decide
whether the family helps or hurts. Rather than guess it, read three points on the curve at once.
"""
import ast, json, os, re, shutil, sys

SRC_NB = "variants/push_legacy4/knee-mri-second-family-four-runs.ipynb"
SRC_META = "variants/push_legacy4/kernel-metadata.json"
share = float(sys.argv[1])
tag = sys.argv[2]
slug = f"knee-mri-second-family-share-{tag}"
out = f"variants/push_share_{tag}"

nb = json.load(open(SRC_NB, encoding="utf-8"))
pat = re.compile(r"LEGACY_MEMBER_WEIGHT_BY_TARGET = \{[^{}]*\}")
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    m = pat.search(j)
    if m:
        d = ast.literal_eval(m.group(0).split("=", 1)[1].strip())
        scaled = {k: round(v * share, 4) for k, v in d.items()}
        j = j[:m.start()] + "LEGACY_MEMBER_WEIGHT_BY_TARGET = " + repr(scaled) + j[m.end():]
        hits += 1
        c["source"] = j.splitlines(keepends=True)
assert hits == 1, f"weight dict matched {hits} times"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))

os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, slug + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open(SRC_META, encoding="utf-8"))
meta["id"] = "dreaddevelopment/" + slug
meta["title"] = f"Knee MRI - Second Family Share {tag}"
meta["code_file"] = slug + ".ipynb"
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print(f"share {share:>4}: {scaled}   -> {out}")
