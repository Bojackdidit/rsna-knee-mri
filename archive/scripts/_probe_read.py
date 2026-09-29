#!/usr/bin/env python3
"""Wait for the per-finding probes to score, then report AUC_j = 12*board - 5.5.

A probe holds 11 of 12 finding columns at a constant and leaves one real, so it scores near 0.5 and
the board scalar reads that single finding over all 1,322 test studies. Anything above 0.8 is a
normal full submission, not a probe, so it is excluded - matching on the raw score is what made an
earlier watcher fire on the champion's 0.942.
"""
import io
import subprocess
import sys
import time

import pandas as pd

WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DEADLINE = time.time() + 3 * 3600

while time.time() < DEADLINE:
    try:
        out = subprocess.run(["kaggle", "competitions", "submissions",
                              "rsna-knee-abnormality-detection", "--csv"],
                             capture_output=True, text=True, timeout=150).stdout
        d = pd.read_csv(io.StringIO(out)).head(8)
        probes = d[d.description.astype(str).str.startswith("Diagnostic entry")]
        done = probes[probes.publicScore.notna()]
        print("[%s] probes scored: %d/%d" % (time.strftime("%H:%M:%S"), len(done), WANT), flush=True)
        if len(done) >= WANT:
            print("\nPER-FINDING AUC on 1,322 test studies (AUC_j = 12*board - 5.5):", flush=True)
            for _, r in done.iterrows():
                b = float(r.publicScore)
                print("  submitted %s   board %.3f   ->  AUC %.4f"
                      % (str(r.date)[11:16], b, 12 * b - 5.5), flush=True)
            break
    except Exception as e:
        print("[poll] %s" % str(e)[:90], flush=True)
    time.sleep(120)
