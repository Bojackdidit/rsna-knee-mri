#!/usr/bin/env python3
"""Build a single-arm 224 kernel that opens only TWO series per study instead of six.

Resolution was never the time lever here. The run opens six DICOM series per study and that is 22.8
of the 29 minutes; the GPU is about 6. So a fast kernel has to cut SERIES, and the arm has to be
trained on the same subset or it is a train/serve mismatch - which is what the accompanying arm
(--slots '0,2', Sagittal-FS + Coronal-FS) is for.

SLOTS_RECOVERED[:2] is SAG_FLUID_FS + COR_FLUID_FS, the same two series as corpus slots 0 and 2.
Between them they carry all twelve findings: sagittal fluid covers nine, coronal fluid supplies the
MCL and the two OA grades that sagittal alone leaves at chance.
"""
import json, os, re, subprocess, sys

SLUG = sys.argv[1]
ARM = sys.argv[2]
DATASET = "dreaddevelopment/raptor-knee-arms-224"
TITLE = sys.argv[3] if len(sys.argv) > 3 else "Knee MRI - Raptor 224 Two Series"
K = sys.argv[4] if len(sys.argv) > 4 else "24"

r = subprocess.run([sys.executable, "make_arms224.py", SLUG, DATASET, ARM,
                    "--title", TITLE], capture_output=True, text=True)
print(r.stdout.strip()[-600:] or r.stderr.strip()[-600:])
d = "variants/push_" + SLUG.replace("-", "_")
nb_path = os.path.join(d, SLUG + ".ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

hit = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code": continue
    src = "".join(cell["source"])
    if "SLOTS = SLOTS_PUBLIC if SLOT_SCHEME" in src and "N_SLOT = len(SLOTS)" in src:
        src = src.replace(
            "N_SLOT = len(SLOTS)",
            "SLOTS = SLOTS[:2]   # two series per study: sagittal fluid + coronal fluid\n"
            "N_SLOT = len(SLOTS)")
        cell["source"] = src.splitlines(keepends=True); hit += 1
if hit != 1:
    print("PATCH FAILED: matched %d cells, expected 1" % hit); sys.exit(1)
json.dump(nb, open(nb_path, "w", encoding="utf-8"))

chk = "".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
print("patched: SLOTS cut to 2 |", "SLOTS = SLOTS[:2]" in chk)
print("kernel dir:", d)
