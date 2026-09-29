#!/usr/bin/env python3
"""Probit fusion: combine the two halves in Gaussian-quantile space instead of raw percentile rank.

The final blend currently converts both halves to percentile ranks and takes a weighted sum:
    out = (1 - w) * rank_pct(transformer) + w * rank_pct(coatnet)
Percentile rank is uniform, so the arithmetic treats the gap between the 50th and 55th percentile as
identical to the gap between the 94th and 99th. For a detection task the tails are where the signal
is, and both halves are far more informative there. Mapping each rank through the Gaussian quantile
(probit) before combining restores that spacing, then the result is re-ranked so the metric is
unaffected by the transform itself.

⚠ Honest status: the public fork carries this and ships NO ablation, so it is an UNTESTED claim, not
a measured gain. It is queued as a board read because our gold-58 gate has mispredicted the board
seven times and cannot settle it offline.

Safe by construction: ranks are clipped away from 0 and 1 before ndtri so no infinities can enter,
the output is re-ranked exactly as before, and the existing finite/range check on the blended values
still runs.
"""
import ast, difflib, json, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_probit"
SLUG = "knee-mri-probit-fusion"
TITLE = "Knee MRI - Probit Fusion"

OLD = """        _blend_output[
            _label
        ] = (
            (
                1.0
                - _cw
            )
            * _blend_tr[
                _label
            ]
            + _cw
            * _blend_cr[
                _label
            ]
        )"""
NEW = """        # Probit fusion: combine in Gaussian-quantile space so tail separation is preserved,
        # instead of adding uniform percentile ranks. Clipped away from 0/1 so ndtri stays finite.
        _eps = 1.0 / (2.0 * max(len(_blend_output), 1))
        _z_tr = _blend_np.clip(_blend_tr[_label].to_numpy(_blend_np.float64), _eps, 1.0 - _eps)
        _z_cr = _blend_np.clip(_blend_cr[_label].to_numpy(_blend_np.float64), _eps, 1.0 - _eps)
        from scipy.special import ndtri as _ndtri
        _blend_output[
            _label
        ] = (
            (
                1.0
                - _cw
            )
            * _ndtri(_z_tr)
            + _cw
            * _ndtri(_z_cr)
        )"""

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits += 1
        c["source"] = j.splitlines(keepends=True)
assert hits == 1, f"blend anchor matched {hits} times (expected 1)"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse; anchor matched exactly once")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True, "must stay private"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print(f"private OK, {len(d)} changed lines -> {OUT_DIR}")
