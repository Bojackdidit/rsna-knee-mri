#!/usr/bin/env python3
"""Concentrate window sampling on the CENTRE of the slice band.

Why: our 20 main members each average all 10 sliding window positions across SLICE_BAND
(0.2-0.8), i.e. "equal-spaced across the range". a published published error
decomposition measured centre-adjacent slices beating equal-spaced by +0.018 macro-AUC on
this competition - the pathology (menisci, ACL, effusion) sits centrally in the sagittal
stack, and the outer windows dilute it.

The kernel ALREADY knows how to do this: when it cannot afford every window it takes the
central n_win positions (`mid = (len(starts_full) - n_win) // 2`). That path only fires
under time pressure, and our run finishes in 3 h 32 m of an 8 h budget, so it never fires
and we always take all 10. This forces the same, already-exercised centring unconditionally.

One variable changes: how many window positions each member averages. Nothing about the
models, the preprocessing, the band, the cache or the blend weight moves - so a board delta
attributes cleanly to window placement alone. Jitter is already on for every member and
stays on (fewer windows only frees budget), so there is no confound from the scheduler.
"""
import ast, difflib, json, os, sys

SRC = "champ/knee-mri-blend-even-full.ipynb"
K = int(sys.argv[1])
OUT_DIR = f"variants/push_center{K}"
SLUG = f"knee-mri-centre-{K}-windows"
TITLE = f"Knee MRI - Centre {K} Windows"

ANCHOR = "        starts_full = window_starts(Cte.shape[2], GROUP)\n"
REPLACE = (
    "        starts_full = window_starts(Cte.shape[2], GROUP)\n"
    f"        # Centre-concentrated window sampling: keep only the middle {K} positions.\n"
    "        # Same centring idiom the affordability path already uses, applied unconditionally.\n"
    f"        if len(starts_full) > {K}:\n"
    f"            _mid = (len(starts_full) - {K}) // 2\n"
    f"            starts_full = starts_full[_mid:_mid + {K}]\n"
    "        log(f'centre-concentrated windows: {len(starts_full)} position(s) -> {starts_full}')\n"
)

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if ANCHOR in j:
        j = j.replace(ANCHOR, REPLACE, 1)
        hits += 1
        c["source"] = j.splitlines(keepends=True)
assert hits == 1, f"anchor matched {hits} times (expected exactly 1)"

for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)

old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
print("\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1)))
print(f"built {OUT_DIR}  (centre {K} of 10 main / 7 legacy windows)")
