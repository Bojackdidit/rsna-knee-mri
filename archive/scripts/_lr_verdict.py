#!/usr/bin/env python3
"""Read the backbone-LR curve and judge it honestly. Usage: _lr_verdict.py LOGDIR

All four runs are CoAtNet-384, k=k_eval=12, grouped fold 0, blend labels, seed 42, 16 epochs, and
differ ONLY in bb_lr. That matters: the grouped gate grades against the `--labels` parquet, so it
measures agreement with a label set - valid here because every run grades against the SAME one, and
invalid the moment a comparison crosses label sets.

Judge on the MATCHED-EPOCH curve, not on the best-so-far. Low LR leads early by construction: it
preserves ImageNet features while a higher LR is still recovering from disturbing them, and under
OneCycleLR the high-LR runs spend the early epochs ramping toward a peak. On efficientnet-b0 the
crossover only appeared by epoch 12. A verdict read before ep12 is reading the ramp, not the result.

Seed noise on this gate is ~0.0032 and the paired minimum-detectable difference is ~0.0053, so a
gap under 0.006 between two arms is not a result.
"""
import os, re, sys, glob

D = sys.argv[1] if len(sys.argv) > 1 else "."
RUNS = [("3e-5 (control)", "run_coat384_ref.log"), ("5e-5", "run_lr5e5.log"),
        ("1e-4", "run_lr1e4.log"), ("3e-4 (killed)", "run_lr3e4.log")]
PAT = re.compile(r"ep(\d+) loss ([0-9.]+) \| GOLD macro-AUC ([0-9.]+)")

curves = {}
for name, f in RUNS:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    curves[name] = {int(e): float(a) for e, _l, a in PAT.findall(txt)}

if not curves:
    sys.exit("no logs found in %s" % D)

eps = sorted(set().union(*[set(c) for c in curves.values()]))
names = [n for n, _ in RUNS if n in curves]
print("%-4s " % "ep" + " ".join("%14s" % n for n in names))
for e in eps:
    print("%-4d " % e + " ".join("%14s" % ("%.4f" % curves[n][e] if e in curves[n] else "-")
                                 for n in names))

print("\nfinal / best per run:")
for n in names:
    c = curves[n]
    last = max(c)
    print("  %-14s epochs %2d | last ep%-2d %.4f | best %.4f" % (n, len(c), last, c[last], max(c.values())))

done = [n for n in names if len(curves[n]) >= 16]
if len(done) < 2:
    print("\nNOT YET JUDGEABLE - %d of %d runs have reached ep15." % (len(done), len(names)))
    print("Everything before ep12 is the OneCycle ramp; do not call it.")
else:
    ranked = sorted(((max(curves[n].values()), n) for n in done), reverse=True)
    print("\nfinished runs, ranked:")
    for v, n in ranked:
        print("   %-14s %.4f" % (n, v))
    top, second = ranked[0], ranked[1]
    gap = top[0] - second[0]
    print("\n  %s beats %s by %.4f" % (top[1], second[1], gap))
    if gap < 0.0053:
        print("  ⛔ BELOW the paired minimum-detectable difference (0.0053). NOT a result.")
    else:
        print("  ✓ exceeds the 0.0053 paired MDD - a real separation.")
    if "3e-5 (control)" in done:
        ctl = max(curves["3e-5 (control)"].values())
        print("\n  vs the inherited 3e-5 control (%.4f):" % ctl)
        for v, n in ranked:
            if n != "3e-5 (control)":
                d = v - ctl
                verdict = "REAL GAIN" if d >= 0.0053 else ("real loss" if d <= -0.0053 else "inside noise")
                print("     %-14s %+.4f   %s" % (n, d, verdict))
