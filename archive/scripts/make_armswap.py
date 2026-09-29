#!/usr/bin/env python3
"""Swap the weaker of our two arms for a better one already published. Same runtime, better gate.

The shipped pair is v11blend_full_swa + confdist_swa. On the 58-study gold gate, scored with the
same equal-weight rank-mean the kernel itself applies:

    v11blend_full_swa   0.9180   <- individually the WEAKEST of the top arms, carried at 50%
    confdist_swa        0.9266
    w80_full_swa        0.9269
    w80_f2_swa          0.9275

    shipped  v11blend + confdist      0.9282
    this     confdist + w80_full      0.9333   (+0.0051, better in 93.5% of 2000 bootstraps,
                                                worst 2.5% -0.0017)

Best of all 36 available pairs. Both weights were already published to Kaggle and neither has ever
been served, so this needs no training. Arm count is unchanged, so the scoring time is unchanged
(the kernel re-preprocesses each study per arm, so time scales with arm count, not arm identity) -
a third arm reaches 0.9340 but costs ~50% more wall clock for +0.0007, which the efficiency metric
prices at 50 s. Weight is left at 0.50/0.50: the profile is flat from 0.45 to 0.60, so the plateau
is entered without fitting anything.

Carries the fast header reads (specific_tags, measured 7.4x, lossless ordering) and skips the
partner half, which measured as dead weight (arm-only 0.941 vs blend 0.940 at 1/6 the passes).
"""
import ast, json, os, sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_armswap"
SLUG = "knee-mri-stronger-arm-pair"
TITLE = "Knee MRI - Stronger Arm Pair"
EXTRA_DATASETS = ["dreaddevelopment/raptor-knee-w80-arm"]

SEED = "\n".join([
    "print('ARMSWAP: partner half not run', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
]) + "\n"

OLD_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')
NEW_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_confdist_swa.pt",          "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_coatnet_w80_full_swa.pt",  "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')

OLD_HDR = "                h = pydicom.dcmread(f, stop_before_pixels=True)"
NEW_HDR = ("                # Parse ONLY the four elements used below. Measured 7.4x faster than a\n"
           "                # full header parse over the same files, with identical ordering.\n"
           "                h = pydicom.dcmread(f, stop_before_pixels=True, force=True,\n"
           "                                    specific_tags=[(0x0020, 0x0032), (0x0020, 0x0037),\n"
           "                                                   (0x0028, 0x0030), (0x0020, 0x0013)])")

EDITS = [
    ("run_dinov2()\n", SEED),
    ('    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n',
     '    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n'
     '    sub.to_csv("/kaggle/working/submission.csv", index=False)\n'
     '    print("ARMSWAP: wrote submission.csv", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
    (OLD_ARMS, NEW_ARMS),
    (OLD_HDR, NEW_HDR),
]

nb = json.load(open(SRC, encoding="utf-8"))
counts = [0]*len(EDITS)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for i,(old,new) in enumerate(EDITS):
        if old in j:
            j = j.replace(old,new,1); counts[i]+=1
    c["source"] = j.splitlines(keepends=True)
assert counts == [1]*len(EDITS), "edits matched %s" % counts
body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
assert "raptor_ft_coatnet_v11blend_full_swa.pt" not in body, "old arm still referenced"
assert body.count("raptor_ft_coatnet_w80_full_swa.pt") == 1
assert body.count("raptor_ft_confdist_swa.pt") == 1
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("champ/kernel-metadata.json",encoding="utf-8")); meta.pop("id_no",None)
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
for d in EXTRA_DATASETS:
    if d not in meta["dataset_sources"]:
        meta["dataset_sources"].append(d)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | arms swapped, v11 gone")
print("2 arms x %d windows = %d passes/study (unchanged); datasets: %s" % (K, 2*K, meta["dataset_sources"]))
