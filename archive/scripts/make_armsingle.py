#!/usr/bin/env python3
"""ARM-ONLY with a SINGLE arm: the leanest thing that can still score.

Two arms of the same family (coatnet_rmlp_2_rw_384) cost double the inference for a same-family
pair. The leaders beat our entire blend with ONE model, so this keeps only the stronger arm
(confdist_swa, gold 0.9266) and drops the other.

  champion   588 passes/study (~3.5 h)
  arm-only    96 passes/study (2 arms x 48)
  this        48 passes/study (1 arm  x 48)  + no partner cache build

Same three structural fixes as the two-arm version: skip the partner at run_dinov2(), seed
submission.csv from ROOT/sample_submission.csv for the V18 block, and have the arm write the final
submission.csv itself.
"""
import ast, json, os, sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_armsingle%d" % K
SLUG = "knee-mri-arm-single-%d" % K
TITLE = "Knee MRI - Arm Single %d" % K

SEED = "\n".join([
    "# ARM-ONLY: partner half not run; V18 downstream still reads submission.csv, so seed it from",
    "# ROOT/sample_submission.csv (ROOT exists at line 29; LAB is defined far later).",
    "print('ARM-ONLY: partner half (20-member DINOv2 package) not run at all', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
    "print('ARM-ONLY: seeded submission.csv', _ao_df.shape, flush=True)",
]) + "\n"

DROP_ARM = '    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'

EDITS = [
    ("run_dinov2()\n", SEED),
    ('    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n',
     '    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n'
     '    sub.to_csv("/kaggle/working/submission.csv", index=False)\n'
     '    print("ARM-ONLY: wrote submission.csv from the arm", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
    (DROP_ARM, "    # single-arm: v11blend dropped, confdist_swa (the stronger arm) kept\n"),
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
assert counts == [1,1,1,1], "edits matched %s" % counts
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("champ/kernel-metadata.json",encoding="utf-8")); meta.pop("id_no",None)
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | passes/study", K)
