#!/usr/bin/env python3
"""Map the legacy-family share curve as soon as the board returns it, and say what to fire next.

Four points, one variable: how much of each finding the second model family carries.
The 0% point is the standing champion at 0.940, already known, so three reads complete the curve.
The 4-model control at 75% separates "the family helps" from "averaging more of it helps".
"""
import csv, io, subprocess, sys
import numpy as np

TAG = {
    "sixteen model family, given a third": ("share 50%", 0.50),
    "again at a ninth":                    ("share 25%", 0.25),
    "second family of models added":       ("share 75%", 0.75),
    "read from only one of the four":      ("4-model control @75%", 0.75),
    "transformer half was reading":        ("denser windows (orthogonal)", None),
}
out = subprocess.run(["kaggle", "competitions", "submissions", "-c",
                      "rsna-knee-abnormality-detection", "--csv"],
                     capture_output=True, text=True, timeout=600).stdout
rows = list(csv.DictReader(io.StringIO(out)))
pts, other = [], []
for r in rows:
    if not r.get("publicScore") or not r["date"].startswith("2026-09-05"):
        continue
    for k, (name, share) in TAG.items():
        if k in r["description"]:
            (pts if share is not None and "control" not in name else other).append(
                (share, float(r["publicScore"]), name))
print("CHAMPION (0% share) = 0.940   [known]")
for s, sc, n in sorted(other): print(f"  {n:28s} {sc:.3f}   ({sc-0.940:+.3f} vs champion)")
if pts:
    print("\nshare curve:")
    for s, sc, n in sorted(pts): print(f"  {s:.0%}  {sc:.3f}   ({sc-0.940:+.3f})")
xs = [0.0] + [p[0] for p in sorted(pts)]
ys = [0.940] + [p[1] for p in sorted(pts)]
if len(xs) >= 3:
    a, b, c = np.polyfit(xs, ys, 2)
    v = -b / (2 * a)
    best = max(zip(ys, xs))
    print(f"\nquadratic fit: optimum share {v:.2f} -> predicted {a*v*v+b*v+c:.4f}")
    print(f"best measured point: {best[0]:.3f} at share {best[1]:.0%}")
    if best[0] <= 0.9405:
        print("VERDICT: family does not beat the champion at any tested share -> axis closed, move on.")
    else:
        print(f"VERDICT: family PAYS. Next fire the fitted share {v:.2f}, then widen which findings it carries.")
else:
    print(f"\nonly {len(xs)-1} of 3 curve points scored so far - waiting.")
