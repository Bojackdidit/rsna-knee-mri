#!/usr/bin/env python3
"""ARM-ONLY on THREE fat-suppressed series. Pairs with an arm trained on corpus slots 0,2,4.

Measured tonight: removing the 20-member partner half cost nothing (0.941 vs 0.940, inside the
+/-0.001 board noise) while cutting 492 passes/study. So the partner is deleted permanently and the
only remaining scoring cost is the ARM's DICOM handling - `order_and_meta` opens every header in
every series to sort slices, 5 series x 1,322 studies, independent of passes or slices kept.

This drops the two non-fat-suppressed series, leaving Sagittal-FS, Coronal-FS and Axial-FS:
    5 series, MAXS 96  ->  3 series, MAXS 63
i.e. 40% fewer series opened. ⚠ It MUST be paired with an arm trained on the matching subset
(--slots 0,2,4); serving a 5-series arm on 3 series is the same train/serve mismatch that killed the
ROI crop.
"""
import ast, json, os, sys

ARM_FILE = sys.argv[1] if len(sys.argv) > 1 else "raptor_ft_coat384_confdist_fs3.pt"
DATASET = sys.argv[2] if len(sys.argv) > 2 else "dreaddevelopment/raptor-knee-fs3-arm"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_armonlyfs3"
SLUG = "knee-mri-arm-only-fs3"
TITLE = "Knee MRI - Arm Only FS3"

SEED = "\n".join([
    "print('ARM-ONLY FS3: partner half not run', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
]) + "\n"

OLD_SLOTS = ('SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),\n'
             '         ("Coronal", 0, 12), ("Axial", -1, 18)]')
NEW_SLOTS = ('# THREE fat-suppressed series only, matching an arm trained on corpus slots 0,2,4.\n'
             '# 40% fewer series opened; order_and_meta reads every header in every series.\n'
             'SLOTS = [("Sagittal", 1, 27), ("Coronal", 1, 18), ("Axial", -1, 18)]')

OLD_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')
NEW_ARMS = ('ARMS = [\n'
            '    # single arm, trained on the same three series this kernel serves\n'
            '    {"file": "%s", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n'
            ']' % ARM_FILE)

EDITS = [
    ("run_dinov2()\n", SEED),
    ('    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n',
     '    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n'
     '    sub.to_csv("/kaggle/working/submission.csv", index=False)\n'
     '    print("ARM-ONLY FS3: wrote submission.csv", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
    (OLD_SLOTS, NEW_SLOTS),
    (OLD_ARMS, NEW_ARMS),
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
print("edits", counts, "| parses | private")
print("series 5->3 (MAXS 96->63), single arm, %d windows -> %d passes/study" % (K, K))
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
