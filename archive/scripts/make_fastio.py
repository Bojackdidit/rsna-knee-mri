#!/usr/bin/env python3
"""ARM-ONLY + fast DICOM header reads. Same predictions, measured 7.4x less header time.

Profiled on real test studies (CPU kernel, no GPU quota):
    header scan  72.9% of IO      pixel decode 12.8%      post-process 14.3%
    full dcmread  9.08s / 557 headers
    specific_tags 1.23s / 557 headers   -> 7.4x, IO/study 3.90s -> 1.28s
The PARTNER path already used specific_tags (lines 244, 277); the ARM path did not - order_and_meta
ran a full header parse on every file in every series just to read four elements.

This is LOSSLESS: same files, same four tags, same sort order. It is not striding ( measured
slice agreement dropping to 39%) and not InstanceNumber (he measured zero speedup because the cost
is opening the file) - he tested WHAT to read; this changes HOW MUCH of each header is parsed.

Built on arm-only, whose predictions are unchanged, so any board delta would be noise and the
scoring time is the measurement.
"""
import ast, json, os, sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_fastio"
SLUG = "knee-mri-fast-header-reads"
TITLE = "Knee MRI - Fast Header Reads"

SEED = "\n".join([
    "print('ARM-ONLY: partner half not run', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
]) + "\n"

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
     '    print("ARM-ONLY: wrote submission.csv", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
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
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | arm-only,", K, "windows, fast headers")
