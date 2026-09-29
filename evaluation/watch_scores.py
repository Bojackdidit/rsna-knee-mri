#!/usr/bin/env python3
"""Emit one line per newly-scored RSNA submission; exit when the 2 newest rows are done."""
import csv, io, subprocess, time

seen = set()
while True:
    r = subprocess.run(["kaggle", "competitions", "submissions",
                        "-c", "rsna-knee-abnormality-detection", "-v"],
                       capture_output=True, text=True)
    try:
        rows = list(csv.DictReader(io.StringIO(r.stdout or "")))[:2]
    except Exception:
        rows = []
    for row in rows:
        ref, status = row.get("ref"), (row.get("status") or "")
        if ref in seen:
            continue
        if "COMPLETE" in status.upper():
            seen.add(ref)
            print("SCORED ref=%s publicScore=%s desc=%s" %
                  (ref, row.get("publicScore"), (row.get("description") or "")[:60]),
                  flush=True)
        elif "ERROR" in status.upper():
            seen.add(ref)
            print("SUBMISSION ERROR ref=%s desc=%s" %
                  (ref, (row.get("description") or "")[:80]), flush=True)
    if rows and len(seen) >= 2:
        print("both scored.", flush=True)
        break
    time.sleep(600)
