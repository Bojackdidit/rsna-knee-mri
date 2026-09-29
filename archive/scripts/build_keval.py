#!/usr/bin/env python3
"""Champion kernel, K_EVAL changed. NOTHING else. Usage: build_keval.py K

Scott scores ~0.949 in ~5 minutes. Our reading alone is 3.7 min, so his model budget is 1-2 min.
That is unreachable by backbone swap - the cheapest backbone we measured still costs 12.2 min at
K_EVAL 48, because 1322 studies x 48 windows = 63,456 forward passes per arm. Model cost is LINEAR
in window count, so the window budget is the only lever that reaches his runtime:

    K=48 -> 15.9 min   K=24 -> 9.8   K=12 -> 6.8   K=9 -> 6.0   K=6 -> 5.2   (effv2s@288 basis)
    on the CHAMPION's CoAtNet basis: K=48 -> 23.8, K=24 -> 13.8, K=12 -> 8.7

I previously recorded this lever as closed. That was wrong on the evidence: both prior "window cut"
entries changed TWO things at once - the 0.890 entry cut SERIES (6->2) and K_EVAL (48->24) together
on a single 224 arm, and the 0.895 entry was an old hour-capped run. Window count has never been
isolated.

And the arms are TRAINED with k=6 windows per study, then inferred at 48. So reducing K_EVAL moves
inference TOWARD the training distribution, not away from it. This changes one number and keeps the
0.941 arms, the 384 resolution, all six series and the fastdcm reader untouched.
"""
import io, json, os, sys, ast, glob

K = int(sys.argv[1])
SRC = "variants/push_fastdcm"
SLUG = "knee-mri-coatnet-384-k%d" % K
TITLE = "Knee MRI - CoAtNet 384 K%d" % K
OUT = "variants/push_" + SLUG.replace("-", "_")

src_nb = glob.glob(os.path.join(SRC, "*.ipynb"))[0]
nb = json.load(io.open(src_nb, encoding="utf-8"))
OLD = "K_EVAL = 48"
NEW = "K_EVAL = %d" % K
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if OLD in s:
        c["source"] = s.replace(OLD, NEW, 1).splitlines(keepends=True); hits += 1
print("K_EVAL hits:", hits)
if hits != 1:
    sys.exit("K_EVAL anchor did not match exactly once")

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for n, cond in [("K_EVAL set",      "K_EVAL = %d" % K in body),
                ("old K_EVAL gone", "K_EVAL = 48" not in body or K == 48),
                ("champion arm 1",  "raptor_ft_confdist_swa.pt" in body),
                ("champion arm 2",  "raptor_ft_coatnet_w80_full_swa.pt" in body),
                ("still 384",       body.count('"res": 384') == 2),
                ("fastdcm intact",  "_fd_read_one" in body),
                ("all six slots",   "SLOTS[:2]" not in body),
                ("no channels_last","channels_last" not in body)]:
    print("  %-18s %s" % (n, "OK" if cond else "FAIL"))
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
