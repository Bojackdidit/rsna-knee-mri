#!/usr/bin/env python3
"""Push the denser-window candidate the moment the weekly GPU quota unfreezes.

The account is frozen: 48.2 h used against a 45 h weekly allowance, refreshing
2026-09-05 00:00 UTC. Nothing can run until then, so this sits on the clock and pushes as
soon as capacity exists rather than losing hours waiting for a human to notice.

It PUSHES only. It never submits - that stays a deliberate act after the run is inspected.
"""
import json, subprocess, sys, time, datetime, os

DIR = r"D:\claude2\main_track\knee\variants\push_dense"
LOG = r"D:\claude2\main_track\knee\fire_dense.log"


def log(m):
    line = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} UTC | {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def quota():
    """(used_hours, allowed_hours, frozen) from the live account, or None if unreadable."""
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


def push():
    r = subprocess.run(["kaggle", "kernels", "push", "-p", DIR],
                       capture_output=True, text=True, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out.strip()


log("waiting for the GPU quota to unfreeze")
while True:
    q = quota()
    if q is not None:
        used, allowed, frozen = q
        if not frozen and used < allowed - 5.0:
            log(f"quota clear: {used:.1f}/{allowed:.1f} h used, not frozen -> pushing")
            rc, out = push()
            log(f"push rc={rc}: {out[-500:]}")
            if rc == 0:
                log("PUSHED. Inspect the run, then submit deliberately.")
                sys.exit(0)
            log("push failed; retrying in 10 min")
            time.sleep(600); continue
        log(f"still blocked: {used:.1f}/{allowed:.1f} h used, frozen={frozen}")
    time.sleep(600)
