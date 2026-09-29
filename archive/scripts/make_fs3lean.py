#!/usr/bin/env python3
"""The efficiency-track kernel: three series, two 3-series arms, and every discarded stage cut.

Same structural cut as make_lean.py - the A5 DINOv3 half and the rad calibration stage both write
/kaggle/working/submission.csv and both are overwritten by the arm before the V18 blend reads it,
so removing them cannot move the score while removing two full per-study DICOM passes.

On top of that this drops the two non-fat-suppressed series. Arms trained on corpus slots 0,2,4
score 0.9211 gold against 0.9224 for the same recipe on all five, so the series are close to free,
and series count is what drives header cost. Aimed at the efficiency prize (-2.22*AUC +
Runtime/32400), where 0.01 AUC is worth 720 s.
"""
import ast, json, os, sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_fs3lean"
SLUG = "knee-mri-three-series-lean"
TITLE = "Knee MRI - Three Series Lean"
EXTRA_DATASETS = ["dreaddevelopment/raptor-knee-fs3-arm"]

SEED = "\n".join([
    "print('FS3LEAN: partner half not run', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
]) + "\n"

STUB2 = ("# A5 DINOv3 5-fold removed: its output is overwritten by the arm before the blend reads it.\n"
         "print('FS3LEAN: A5 transformer half not run', flush=True)\n")
STUB3 = ("# Calibration stage removed for the same reason; it also re-read every DICOM header.\n"
         "print('FS3LEAN: rad calibration not run', flush=True)\n")

OLD_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')
NEW_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_coat384_confdist_fs3.pt",  "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_coat384_confdist_fs3b.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')
OLD_SLOTS = ('SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),\n'
             '         ("Coronal", 0, 12), ("Axial", -1, 18)]')
NEW_SLOTS = ('# THREE fat-suppressed series (matches arms trained on corpus slots 0,2,4).\n'
             'SLOTS = [("Sagittal", 1, 27), ("Coronal", 1, 18), ("Axial", -1, 18)]')
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
     '    print("FS3LEAN: wrote submission.csv", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
    (OLD_ARMS, NEW_ARMS),
    (OLD_SLOTS, NEW_SLOTS),
    (OLD_HDR, NEW_HDR),
]

nb = json.load(open(SRC, encoding="utf-8"))
a5 = [i for i,c in enumerate(nb["cells"]) if c["cell_type"]=="code"
      and "_A5_SAVED" in "".join(c["source"]) and "A5_W" in "".join(c["source"])]
rad = [i for i,c in enumerate(nb["cells"]) if c["cell_type"]=="code"
       and "_rad" + "_main()" in "".join(c["source"])]
assert len(a5)==1 and len(rad)==1 and a5[0]!=rad[0], "stage identification failed"
assert "run_dinov2()" not in "".join(nb["cells"][a5[0]]["source"])
assert "ARMS = [" not in "".join(nb["cells"][rad[0]]["source"])
nb["cells"][a5[0]]["source"] = STUB2.splitlines(keepends=True)
nb["cells"][rad[0]]["source"] = STUB3.splitlines(keepends=True)
print("cut cell %d (A5) and cell %d (rad)" % (a5[0], rad[0]))

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
assert "raptor_ft_coatnet_v11blend_full_swa.pt" not in body
assert '("Coronal", 0, 12)' not in body, "still on five series"
assert body.count("raptor_ft_coat384_confdist_fs3.pt") == 1
assert body.count("raptor_ft_coat384_confdist_fs3b.pt") == 1
assert "_A5_SAVED" not in body and "_rad" + "_main" not in body, "a cut stage survived"
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
print("edits", counts, "| parses | private | 3 series, 2 fs3 arms, A5+rad cut")
