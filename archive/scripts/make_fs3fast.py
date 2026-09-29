#!/usr/bin/env python3
"""TWO 3-series arms + fast DICOM header reads. Both time levers in one kernel.

Composes the two measured wins:
  1. partner half removed + 3 fat-suppressed series (make_fs3pair.py) - series count is the only
     driver of header cost, and 3-series training == 5-series on gold (0.9211 vs 0.9224).
  2. specific_tags header parse (make_fastio.py) - 7.4x on the same files, lossless ordering.
Neither changes what the models see, so the AUC should track the 0.941 arm-only read.
"""
import ast, json, os, sys

ARM_A = sys.argv[1]
ARM_B = sys.argv[2]
DATASET = sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 48

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_fs3fast"
SLUG = "knee-mri-fs3-fast-headers"
TITLE = "Knee MRI - Three Series With Fast Header Reads"

SEED = "\n".join([
    "print('FS3FAST: partner half not run', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
]) + "\n"

OLD_SLOTS = ('SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),\n'
             '         ("Coronal", 0, 12), ("Axial", -1, 18)]')
NEW_SLOTS = ('# THREE fat-suppressed series (matches arms trained on corpus slots 0,2,4).\n'
             'SLOTS = [("Sagittal", 1, 27), ("Coronal", 1, 18), ("Axial", -1, 18)]')

OLD_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')
ROW = '    {"file": "%s", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},'
NEW_ARMS = "ARMS = [\n" + (ROW % ARM_A) + "\n" + (ROW % ARM_B) + "\n]"

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
     '    print("FS3FAST: wrote submission.csv", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
    (OLD_SLOTS, NEW_SLOTS),
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
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("champ/kernel-metadata.json",encoding="utf-8")); meta.pop("id_no",None)
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
if DATASET not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DATASET)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private")
print("3 series + fast headers, 2 arms x %d windows = %d passes/study (champion: 588)" % (K, 2*K))
