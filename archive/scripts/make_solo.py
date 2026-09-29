#!/usr/bin/env python3
"""SOLO-arm kernel: one model, one fold, nothing blended into it.

Built on the SAME base as make_newarm.py - variants/push_twogpu2/knee-mri-dual-gpu-decode.ipynb,
the 0.941 @ 29.1 min kernel - so every speed win is retained: two dead prediction stages cut, one
DICOM pass instead of one per arm, threaded header reads, threaded pixel decode, reads overlapped
with inference.

Why the wall clock does not get worse by dropping an arm: the two shipped arms run CONCURRENTLY,
one per GPU, so the run time is max(arm0, arm1), not the sum. Removing one leaves the other on its
own GPU doing exactly what it did before. DICOM reading is ~73% of the clock and is unchanged.
So this lands at or under the 29-minute ceiling, never over it.

The single arm carries w=1.0 - with one entry the weighted mean is just that arm's probabilities.

Usage:
    python make_solo.py <arm_filename.pt> <kaggle-dataset-slug> [K_EVAL]
"""
import ast, json, os, sys

ARM = sys.argv[1]
DATASET = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 48

SRC = "variants/push_twogpu2/knee-mri-dual-gpu-decode.ipynb"
OUT_DIR = "variants/push_solo"
SLUG = "knee-mri-raptor-solo-arm"
TITLE = "Knee MRI - Raptor Solo Arm"

OLD = ('    {"file": "raptor_ft_confdist_swa.pt",          '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
       '    {"file": "raptor_ft_coatnet_w80_full_swa.pt",  '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},')
NEW = ('    {"file": "%s", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},'
       % ARM)

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == 1, "two-arm block matched %d times (expected 1)" % hits

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
# exactly one arm, and both shipped arms gone
assert body.count(ARM) == 1, "solo arm not wired exactly once"
assert "raptor_ft_confdist_swa.pt" not in body, "shipped arm 0 still referenced"
assert "raptor_ft_coatnet_w80_full_swa.pt" not in body, "shipped arm 1 still referenced"
assert body.count('"w": 1.0') == 1 and '"w": 0.5' not in body, "weights not solo"
# every speed win must still be present
for marker in ("_hdr_one", "_px_one", "_prep_study", "_PARALLEL_ARMS", "[1pass]"):
    assert marker in body, "lost optimisation: %s" % marker
assert "_A5_SAVED" not in body and "_rad" + "_main" not in body, "a cut stage came back"
assert "K_EVAL = %d" % K in body, "K_EVAL is not %d" % K
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations",
                                               "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("variants/push_twogpu2/kernel-metadata.json", encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
if DATASET not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DATASET)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("built %s" % OUT_DIR)
print("  SOLO arm -> %s   (w=1.0, both shipped arms removed)" % ARM)
print("  all speed optimisations verified present; K_EVAL %d" % K)
print("  datasets: %s" % meta["dataset_sources"])
