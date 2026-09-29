#!/usr/bin/env python3
"""FAST kernel: cut the work on BOTH halves at once, using the one lever with a measured ladder.

Cost accounting per study, as the champion runs today:
    partner  20 members x 10 windows x 2 views (jitter) = 400 passes
    our arm   94 windows x 2 arms                        = 188 passes
    total                                                ~ 588 passes  -> ~3.5 h over 1,322 studies
The leaders score HIGHER than us in 5-25 minutes, so this volume is not buying its keep. Our own
published reading says 42 arm windows scored 0.930 and 62 bought only +0.003, i.e. the sampling is
deep into diminishing returns.

This variant cuts both halves to centre-concentrated windows:
    partner  20 x 4 x 2 = 160
    our arm  48 x 2     =  96
    total               ~ 256 passes  -> about 2.3x less work
and the cut is not arbitrary - it is a published measured ladder (24-spread 0.7695 ->
12-adjacent 0.7777 -> centre-9 0.7815, "slice geometry mattered more than resolution"), so it may
raise the score rather than merely cost less.

The arm side is centred PER SEQUENCE. Our arm's volume is five series concatenated (Sagittal-FS 27,
Sagittal 21, Coronal-FS 18, Coronal 12, Axial 18 = 96), so a global centre cut would quietly become
"keep coronal, drop sagittal". Each series keeps its own middle half instead.
"""
import ast, difflib, json, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_fast"
SLUG = "knee-mri-fast-centre-both-halves"
TITLE = "Knee MRI - Fast Centre Both Halves"
PARTNER_K, ARM_K, FRAC = 4, 48, 0.5

HELPER = f'''
# --- per-slot centre concentration for the arm ------------------------------------------------
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
OLD_PART = "        starts_full = window_starts(Cte.shape[2], GROUP)\n"
NEW_PART = (
    "        starts_full = window_starts(Cte.shape[2], GROUP)\n"
    f"        if len(starts_full) > {PARTNER_K}:\n"
    f"            _mid = (len(starts_full) - {PARTNER_K}) // 2\n"
    f"            starts_full = starts_full[_mid:_mid + {PARTNER_K}]\n"
    "        log(f'FAST: partner windows -> {len(starts_full)} centre position(s) {starts_full}')\n")

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"helper": 0, "part": 0, "k": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD_EW in j:
        j = j.replace(OLD_EW, HELPER.strip("\n") + "\n\n" + NEW_EW, 1); hits["helper"] += 1
    if OLD_PART in j:
        j = j.replace(OLD_PART, NEW_PART, 1); hits["part"] += 1
    if "K_EVAL = 94" in j:
        j = j.replace("K_EVAL = 94", f"K_EVAL = {ARM_K}", 1); hits["k"] += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == {"helper": 1, "part": 1, "k": 1}, hits
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
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print("all cells parse; anchors:", hits)
print(f"partner 10->{PARTNER_K} windows, arm 94->{ARM_K} per-slot-centred, {len(d)} changed lines")
print("passes/study: 588 -> %d  (%.1fx less work)" % (20*PARTNER_K*2 + ARM_K*2, 588/(20*PARTNER_K*2 + ARM_K*2)))
