#!/usr/bin/env python3
"""Poll one RSNA submission to a score and record when it lands.

Runtime is only observable while the submission is in flight, so this starts with the submit
rather than after it - see the rsna-runtime-poll-before-submit lesson.
"""
import io, subprocess, sys, time
import pandas as pd

REF = int(sys.argv[1])
t0 = time.time()
while True:
    try:
        out = subprocess.run(["kaggle", "competitions", "submissions",
                              "rsna-knee-abnormality-detection", "--csv"],
                             capture_output=True, text=True, timeout=120).stdout
        d = pd.read_csv(io.StringIO(out))
        r = d[d.ref == REF]
        if len(r):
            r = r.iloc[0]
            print("[%5.1f min] %s  score=%s" % ((time.time() - t0) / 60, r.status, r.publicScore), flush=True)
            if "COMPLETE" in str(r.status) or "ERROR" in str(r.status):
                break
    except Exception as e:
        print("[poll] %s" % str(e)[:80], flush=True)
    time.sleep(90)
