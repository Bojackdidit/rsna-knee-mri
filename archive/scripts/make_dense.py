#!/usr/bin/env python3
"""Build the 'denser partner windows' variant from the standing 0.940 champion.

ONE variable changes. The 20-member partner stack currently scores each study at 10 window
positions, because the weights package records 12 cached slices per slot and a window is 3
slices (12 - 3 + 1 = 10). Our own measurement of the live run is 3 h 32 m against a TIME_BUDGET
of 8 h - 78 min for our arm, 134 min for the partner stack - so more than half the inference
budget goes unused every submission.

This spends part of it on the half that carries the score: the same anatomical band is sampled
at a higher slice density, which yields more window positions for the members to average over.
The members take 3-slice windows, so nothing about their input changes; they simply vote more
times, over positions that already lie inside the band they were fitted on.

Guarded three ways so it can never be the reason a run dies:
  1. the extra slices are only taken if they fit the kernel's OWN cache budget
     (CACHE_FRACTION of MemAvailable, hard-capped at CACHE_BUDGET_MAX_GB) measured live;
  2. on any shortfall it falls back to exactly the champion's value, i.e. this run;
  3. the existing per-member scheduler still trims windows and drops jitter against the clock,
     so a slower run degrades gracefully instead of timing out.
"""
import json, re, sys, shutil, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_dense"
DENSE = int(sys.argv[1]) if len(sys.argv) > 1 else 20
FRAC  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.62
FLOOR = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
SLUG = f"knee-mri-partner-dense-{DENSE}"
TITLE = f"Knee MRI Partner Windows {DENSE}"

nb = json.load(open(SRC, encoding="utf-8"))

ANCHOR = "    CACHE_SLICES = int(cfg['slices'])\n"
PATCH = (
    "    CACHE_SLICES = int(cfg['slices'])\n"
    "    # --- denser TTA windows for the partner members (the ONE change vs the 0.940 run) ---\n"
    "    # Same band, same 3-slice windows, more positions to average over. Our own timing of the\n"
    "    # live run is 3 h 32 m against TIME_BUDGET 8 h, so this spends idle budget on the half\n"
    "    # that carries the score. Taken only if it fits memory AND leaves the floor free.\n"
    f"    _want = {DENSE}\n"
    f"    _frac, _floor = {FRAC}, {FLOOR}\n"
    "    if _want > CACHE_SLICES:\n"
    "        try:\n"
    "            _n_test = len(pd.read_csv(ROOT / 'test.csv'))\n"
    "            _avail = available_gb()\n"
    "            _per = max(_n_test, 1) * N_SLOT * CACHE_IMG * CACHE_IMG / 1024 ** 3\n"
    "            _cap = min(_avail * _frac, _avail - _floor, CACHE_BUDGET_MAX_GB)\n"
    "            _afford = int(_cap // max(_per, 1e-9))\n"
    "            _use = max(CACHE_SLICES, min(_want, _afford))\n"
    "            log(f'partner windows: cfg {CACHE_SLICES} slices/slot; {_n_test} test studies, '\n"
    "                f'{_avail:.1f} GB avail, {_per * _use:.1f} GB cache -> using {_use} slices '\n"
    "                f'= {_use - GROUP + 1} window positions (was {CACHE_SLICES - GROUP + 1})')\n"
    "            CACHE_SLICES = _use\n"
    "        except Exception as _exc:\n"
    "            log(f'partner windows: densification skipped ({type(_exc).__name__}: {_exc})')\n"
)

hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = c["source"]
    joined = "".join(src)
    if ANCHOR in joined:
        joined = joined.replace(ANCHOR, PATCH, 1)
        hits += 1
        c["source"] = joined.splitlines(keepends=True)
assert hits == 1, f"anchor matched {hits} times, expected exactly 1"

os.makedirs(OUT_DIR, exist_ok=True)
nb_path = os.path.join(OUT_DIR, SLUG + ".ipynb")
json.dump(nb, open(nb_path, "w", encoding="utf-8"))

meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)

# proof the change is exactly what is claimed
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
import difflib
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1)]
print("\n".join(d))
print(f"\nadded {sum(1 for l in d if l.startswith('+') and not l.startswith('+++'))} lines, "
      f"removed {sum(1 for l in d if l.startswith('-') and not l.startswith('---'))}")
print("wrote", nb_path)
