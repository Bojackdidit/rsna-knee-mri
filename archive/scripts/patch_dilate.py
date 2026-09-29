#!/usr/bin/env python3
"""Match the serve-time window THICKNESS to what the model was trained on.
Usage: patch_dilate.py SRC.ipynb OUT.ipynb D

A window is three slices shown to the model as RGB. In training they are three adjacent slices of a
44-slice corpus; at serve they are three adjacent slices of a 96-slice volume. Same three slices,
very different amounts of anatomy:

    series      train  serve   train span  serve span  ratio
    Sag-FS         12     27       25.0%       11.1%    2.25x
    Cor-FS          8     18       37.5%       16.7%    2.25x
    Axial           8     18       37.5%       16.7%    2.25x

So every window of every submission we have ever made covers less than half the anatomy the model
learned on. Dilating the serve window closes it:

    d=1  ->  3 of 96  = 11.1% of a 27-slice series   (what we ship today)
    d=2  ->  5 of 96  = 18.5%
    d=3  ->  7 of 96  = 25.9%   <- training span was 25.0%

Centre unchanged, window count unchanged, cost unchanged - only the two outer slices move outward.
Cannot be gated offline: our gate runs on the 44-slice corpus where train and eval already match, so
this defect is invisible there and only the board can see it.
"""
import io
import json
import sys

SRC, OUT, D = sys.argv[1], sys.argv[2], int(sys.argv[3])
OLD = "tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0"


def replace(src, d):
    """Rewrite the window construction, preserving whatever indentation the site actually has."""
    out, hits = [], 0
    for line in src.split("\n"):
        if OLD in line:
            ind = line[:len(line) - len(line.lstrip())]
            out.append(ind + "_d = %d  # serve dilation: match the training window thickness" % d)
            out.append(ind + "_lo = max(0, c - _d); _hi = min(vol.shape[0] - 1, c + _d)")
            out.append(ind + "tri = np.stack([vol[_lo], vol[c], vol[_hi]], 0)"
                             ".astype(np.float32) / 255.0")
            hits += 1
        else:
            out.append(line)
    return "\n".join(out), hits


nb = json.load(io.open(SRC, encoding="utf-8"))
total = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if OLD not in src:
        continue
    src, h = replace(src, D)
    total += h
    lines = [l + "\n" for l in src.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    cell["source"] = lines

if total != 1:
    sys.exit("expected exactly one window construction, found %d" % total)
json.dump(nb, io.open(OUT, "w", encoding="utf-8"), indent=1)
print("patched %d site -> %s (dilation %d)" % (total, OUT, D))
