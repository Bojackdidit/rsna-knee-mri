#!/usr/bin/env python3
"""Swap a newly trained arm into the shipped kernel, in place of the WEAKER of the two.

Built on the current best kernel (dreaddevelopment/knee-mri-dual-gpu-decode, 0.941 @ 29.1 min)
so every speed win is retained: two dead prediction stages cut, one DICOM pass instead of one per
arm, threaded header reads, threaded pixel decode, reads overlapped with inference, and one arm per
GPU. Arm COUNT is unchanged - two arms, one per T4 - so the 29-minute ceiling holds and the only
thing that moves is which weights arm 0 loads.

The shipped pair is confdist_swa (0.9266 gold-58) + w80_full_swa (0.9269). confdist_swa is the
weaker, so that is the one replaced.

Usage:
    python make_newarm.py <new_arm_filename.pt> <kaggle-dataset-slug> [K_EVAL]
"""
import ast, json, os, sys

NEW_ARM = sys.argv[1]
DATASET = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 48

SRC = "variants/push_twogpu2/knee-mri-dual-gpu-decode.ipynb"
OUT_DIR = "variants/push_newarm"
SLUG = "knee-mri-raptor-full-corpus"
TITLE = "Knee MRI - Raptor Full Corpus"

OLD = ('    {"file": "raptor_ft_confdist_swa.pt",          '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},')
NEW = ('    {"file": "%s", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},'
       % NEW_ARM)

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == 1, "arm line matched %d times" % hits

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
# the new arm replaces confdist; the other shipped arm must survive untouched
assert body.count(NEW_ARM) == 1, "new arm not wired exactly once"
assert "raptor_ft_confdist_swa.pt" not in body, "old arm still referenced"
assert body.count("raptor_ft_coatnet_w80_full_swa.pt") == 1, "second arm lost"
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
print("  arm 0 -> %s   (replaces confdist_swa, the weaker shipped arm)" % NEW_ARM)
print("  arm 1 -> raptor_ft_coatnet_w80_full_swa.pt  (unchanged)")
print("  all speed optimisations verified present; 2 arms, one per GPU; K_EVAL %d" % K)
print("  datasets: %s" % meta["dataset_sources"])
