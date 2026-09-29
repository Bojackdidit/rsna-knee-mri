#!/usr/bin/env python3
"""Push the queued kernels the moment the weekly GPU quota resets, in priority order.

The account is at `Maximum weekly GPU quota of 30.00 hours reached`, so nothing can run until the
reset. This sits on the queue and pushes as capacity appears, so no time is lost waiting for a human
to notice. It pushes ONLY - it never submits. Submissions stay a human decision.

BUDGET, so the reset is not squandered the way the last one was:
    RSNA blend run   ~3.5 h each  x2 = 7.0 h
    biocell tracking ~0.6 h each  x3 = 1.8 h
                                    = 8.8 h of the 30 h week
leaving room for a follow-up once the first results are read.

PRIORITY
  1. v11 @ 0.50 - the best arm we have (gold OOF 0.9176 vs v9blend's 0.9120), same blend weight
     the board already confirmed optimal in both directions on the v9 arm.
  2. v11 @ 0.55 - the weight was tuned against the WEAKER v9 arm (0.50 -> 0.938, 0.60 -> 0.937).
     A stronger arm plausibly deserves more weight, and that has never been tested with this arm.
  3. biocell detection variants - the 3 um NMS pool radius against a 3.49 um miss residual, cheap.

Two errors mean different things and must not be confused:
  "Maximum batch GPU session count of 2" = concurrency, wait and retry
  "Maximum weekly GPU quota"             = the week is gone, keep waiting
"""
import os
import subprocess
import sys
import time

# Durable copies on D:. The originals live in the session scratchpad under
# C:\Users\...\Temp\claude\..., which does not survive a reboot - so point at these instead.
KERN = r"D:\claude2\main_track\knee\phase2\blend_variants"
BIO = r"D:\claude2\main_track\biocell\variants"

# Dread 2026-08-26: fire ONLY the best two at reset - the biocell variants are held back
# so the week is not spent before the v11 results are read.
QUEUE = [
    ("RSNA v11 @ w0.50", os.path.join(KERN, "push_v11")),
    ("RSNA v11 @ w0.55", os.path.join(KERN, "push_v11_w55")),
]


def push(path):
    r = subprocess.run(["kaggle", "kernels", "push", "-p", path],
                       capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def main():
    deadline = time.time() + 14 * 3600
    pending = [(n, p) for n, p in QUEUE if os.path.isdir(p)]
    for n, p in QUEUE:
        if not os.path.isdir(p):
            print("MISSING, skipped: %s (%s)" % (n, p), flush=True)
    print("queued: %d" % len(pending), flush=True)

    while pending and time.time() < deadline:
        name, path = pending[0]
        out = push(path)
        if "successfully pushed" in out:
            print("[%s] PUSHED  %s" % (time.strftime("%H:%M:%S"), name), flush=True)
            pending.pop(0)
            time.sleep(20)
            continue
        if "Maximum batch GPU session count" in out:
            print("[%s] slots full, %d still queued" % (time.strftime("%H:%M:%S"), len(pending)),
                  flush=True)
        elif "weekly GPU quota" in out:
            print("[%s] quota still exhausted, waiting" % time.strftime("%H:%M:%S"), flush=True)
        else:
            print("[%s] %s -> %s" % (time.strftime("%H:%M:%S"), name,
                                     out.strip().splitlines()[-1:] or "?"), flush=True)
        time.sleep(180)

    print("done. remaining unpushed: %s" % [n for n, _ in pending], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
