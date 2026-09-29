#!/usr/bin/env python3
"""ARM-ONLY kernel: drop the 20-member partner half entirely.

Our single CoAtNet arm scored 0.936-0.938 on the board alone vs 0.940 for the full blend, so the
partner half's cache build and ~400 passes/study buy ~0.002-0.004. Removing it removes most of the
decode AND most of the inference - the only route to the one-hour bar that keeps the model doing the
real work.

Two failures found and fixed along the way:
  1. Emptying the partner's decode loop trips its own guard, "no member produced predictions".
     It must be skipped at its entry point, run_dinov2().
  2. The V18 block downstream still reads /kaggle/working/submission.csv, so skipping the partner
     leaves it missing. It is seeded from sample_submission.csv and the arm overwrites it.
"""
import ast, json, os, sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_armonly%d" % K
SLUG = "knee-mri-arm-only-%d" % K
TITLE = "Knee MRI - Arm Only %d" % K

SEED_LINES = [
    "# ARM-ONLY: partner half not run. The V18 block downstream still reads submission.csv, so seed",
    "# it from the competition's own sample_submission via ROOT (defined line 29, before this point;",
    "# LAB is not defined until much later, and a /kaggle/input/*/ glob does not match this mount).",
    "print('ARM-ONLY: partner half (20-member DINOv2 package) not run at all', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
    "print('ARM-ONLY: seeded submission.csv', _ao_df.shape, flush=True)",
]
E1_OLD = "run_dinov2()\n"
E1_NEW = "\n".join(SEED_LINES) + "\n"

E2_OLD = '    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n'
E2_NEW = ('    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n'
          '    # ARM-ONLY: the partner half never ran, so our arm IS the submission.\n'
          '    sub.to_csv("/kaggle/working/submission.csv", index=False)\n'
          '    print("ARM-ONLY: wrote submission.csv from the arm", flush=True)\n')

E3_OLD = "K_EVAL = 94"
E3_NEW = "K_EVAL = %d" % K

nb = json.load(open(SRC, encoding="utf-8"))
counts = [0, 0, 0]
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for i, (old, new) in enumerate(((E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (E3_OLD, E3_NEW))):
        if old in j:
            j = j.replace(old, new, 1)
            counts[i] += 1
    c["source"] = j.splitlines(keepends=True)
assert counts == [1, 1, 1], "edits matched %s" % counts
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("edits", counts, "| parses | private | passes/study", 2 * K)
