#!/usr/bin/env python3
"""Push both RSNA candidates the moment the weekly GPU quota unfreezes.

The account is frozen (48.21 h used against a 45.00 h weekly allowance, refresh 2026-09-05
00:00 UTC), so nothing can run until then. This sits on the clock so no time is lost.

Two independent single-variable changes off the standing 0.940 champion, both spending inference
budget our own logs prove is idle (3 h 32 m used of an 8 h TIME_BUDGET):
  1. denser window sampling  - more window positions for the partner members. MEMORY bound.
  2. more augmented views    - more views per window for the same members. TIME bound, no memory.
They are not confounded with each other; together they close the partner-half TTA axis in one day
instead of two. Kaggle allows two concurrent GPU sessions, so both go at once.

It PUSHES only. Submitting stays a deliberate act after each run is inspected.
"""
import datetime, subprocess, sys, time

QUEUE = [
    ("denser window sampling", r"D:\claude2\main_track\knee\variants\push_dense"),
    ("more augmented views",   r"D:\claude2\main_track\knee\variants\push_views"),
]
LOG = r"D:\claude2\main_track\knee\fire_queue2.log"
RESERVE_H = 5.0          # do not start if under this much of the weekly allowance is left


def log(m):
    line = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} UTC | {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def quota():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        a = KaggleApi(); a.authenticate()
        g = a.quota_view().gpu_quota
        return (g.time_used.total_seconds() / 3600.0,
                g.total_time_allowed.total_seconds() / 3600.0,
                bool(getattr(g, "_is_frozen", False)))
    except Exception as exc:
        log(f"quota read failed ({type(exc).__name__}: {exc})")
        return None


log(f"armed: {len(QUEUE)} candidate(s) waiting on the quota reset")
pending = list(QUEUE)
while pending:
    q = quota()
    if q is None:
        time.sleep(600); continue
    used, allowed, frozen = q
    if frozen or used > allowed - RESERVE_H:
        log(f"blocked: {used:.1f}/{allowed:.1f} h used, frozen={frozen}")
        time.sleep(600); continue
    name, d = pending[0]
    log(f"quota clear ({used:.1f}/{allowed:.1f} h) -> pushing '{name}'")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                       capture_output=True, text=True, timeout=1800)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode == 0 and "error" not in out.lower():
        log(f"PUSHED '{name}': {out[-300:]}")
        pending.pop(0)
        time.sleep(120)          # let the first session claim its GPU before the second
    else:
        log(f"push FAILED for '{name}' rc={r.returncode}: {out[-400:]}; retry in 10 min")
        time.sleep(600)
log("all candidates pushed. Inspect each run, then submit deliberately.")
