#!/usr/bin/env python3
"""Per-slot centre-concentrated windows for OUR ARM, with far fewer of them.

Two things this fixes at once - accuracy and runtime - which is the only direction worth taking
when the leaders score higher than us in 5 minutes.

1. COVERAGE GAP. The centre-window variants built earlier patch `starts_full`, which is the PARTNER
   half's path. Our arm does not use it: it calls eval_windows -> _eval_centers, which linspaces
   across the whole valid range. So the arm has never been centre-concentrated at all.

2. GLOBAL CENTRING WOULD BE WRONG HERE. Our arm's volume is five SEQUENCES concatenated
   (Sagittal-FS 27, Sagittal 21, Coronal-FS 18, Coronal 12, Axial 18 = 96). The middle of that
   stack is coronal, so a global centre cut would silently become "keep coronal, drop sagittal",
   not "keep the middle of each series". a published centre-9 result was WITHIN one sequence.
   This restricts to the central fraction of EACH slot's own index range, which is the anatomical
   version of that finding.

3. RUNTIME. K_EVAL is 94 windows per study PER ARM (188 model passes per study across two arms).
   Our own published measurement is 42 windows -> 0.930 and 62 -> +0.003, i.e. heavily diminishing.
   This drops to K_EVAL windows drawn from the central half of each series.

Failure direction is safe: if the restriction leaves too few valid slices the original mask is used.
"""
import ast, difflib, json, os, sys

K_EVAL_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 48
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = f"variants/push_armcentre{K_EVAL_NEW}"
SLUG = f"knee-mri-arm-per-slot-centre-{K_EVAL_NEW}"
TITLE = f"Knee MRI - Arm Per Slot Centre {K_EVAL_NEW}"

HELPER = f'''
# --- per-slot centre concentration for the arm ------------------------------------------------
# build_study fills `vol` slot by slot in SLOTS order (k slices each), so each sequence occupies a
# known contiguous index range. Keeping the middle fraction of EACH range concentrates within every
# series, instead of collapsing onto whichever series happens to sit mid-stack.
ARM_CENTRE_FRAC = {FRAC}

def _per_slot_centre_mask(D):
    keep = np.zeros(D, bool)
    i = 0
    for _name, _fluid, k in SLOTS:
        a, b = i, min(i + k, D)
        if b > a:
            n = b - a
            m = max(3, int(round(n * ARM_CENTRE_FRAC)))
            lo = a + (n - m) // 2
            keep[lo:min(lo + m, b)] = True
        i += k
    return keep

'''

OLD_EW = """def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)"""
NEW_EW = """def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    _m = np.asarray(mask).astype(bool) & _per_slot_centre_mask(D)
    if _m.sum() >= 3:
        mask = _m.astype(np.uint8)
    cs = _eval_centers(mask, D, k)"""

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"helper": 0, "ew": 0, "k": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD_EW in j:
        j = j.replace(OLD_EW, HELPER.strip("\n") + "\n\n" + NEW_EW, 1)
        hits["helper"] += 1; hits["ew"] += 1
    if "K_EVAL = 94" in j:
        j = j.replace("K_EVAL = 94", f"K_EVAL = {K_EVAL_NEW}", 1); hits["k"] += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == {"helper": 1, "ew": 1, "k": 1}, hits
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse; anchors:", hits)

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print(f"K_EVAL 94 -> {K_EVAL_NEW}, per-slot centre frac {FRAC}, {len(d)} changed lines -> {OUT_DIR}")
