#!/usr/bin/env python3
"""RSNA: push, verify and submit the queued candidates as soon as the GPU quota returns.

The account is frozen (48.21 h used of a 45.00 h weekly allowance; refresh 2026-09-05 00:00 UTC),
so nothing can run until then. This waits on the live quota, then works the queue unattended.

All four candidates spend inference budget our own logs prove is idle: the blend scores the hidden
test in 3 h 32 m against a TIME_BUDGET of 8 h. Each is a single-variable change off the standing
0.940 champion `knee-mri-blend-even-full`, so a board delta attributes cleanly.

  1. restored second model family - the biggest lever. Our reproduction stubbed
     `legacy_group_members()` to `return {}`; the public cross-team notebooks use it to load four
     DINOv2 fold models at a different geometry (224 px, 160 mm crop, band 0.2-0.8, legacy pixel
     rules) as their own decode group. They carry weight on only four findings - Lateral Meniscus
     15.0, Lateral OA 15.0, Contusion 5.0, Medial OA 2.5 - which are almost exactly the findings our
     per-finding OOF says we are weakest at. Real trained models where we actually lose.
  2. second model family, four runs - tonylica publishes FOUR such runs (v1-v4, 16 fold models in
     total, genuinely different: fold scores 0.72-0.84) and every public kernel, and ours, loads
     only v1. This loads all four and divides the per-member target weights by four, so the family
     gets exactly the same share of each finding but is estimated from 16 models instead of 4. They
     share a geometry, so it costs one decode group and twelve extra forwards.
  3. denser window sampling - the partner's 20 members score 10 window positions each; this raises
     the slice cache so they get 18. Memory bound, guarded, falls back to champion behaviour.
  4. more augmented views - 3 views per jittered window instead of 2. Time bound, zero memory, so
     it uses budget that (2) cannot.

Kaggle runs at most two GPU sessions at once, so the queue keeps two in flight and starts the third
as slots free. Every run is verified against its own log before a submission slot is spent: a
broken run costs nothing, an unverified submit costs a slot we only get five of.
"""
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "rsna-knee-abnormality-detection"
ARM = "raptor_ft_coatnet_v11blend_full_swa"
LOG = os.path.join(HERE, "rsna_campaign.log")
RESERVE_H = 6.0          # never start a run without this much weekly allowance left
MAX_INFLIGHT = 2

CANDIDATES = [
    # FINAL ORDER 2026-09-05 21:05 UTC, after reading the forum AND the public stack's source.
    #
    # ⛔ DROPPED w=0.85 and the two-group per-finding weights. The public 0.936 stack decomposes as
    # DINOv2 20-member 0.899 -> +DINOv3 0.910 -> +RadImageNet 0.920 -> +our CoAtNet arms 0.936.
    # So the partner half alone is 0.920 ON THE BOARD, not the 0.8375 the 58-study gold gate
    # reported, and our own arm alone is 0.914-0.924 (our public docstring: 0.924 single model,
    # 0.9167 gold). Both halves are comparable, the blend genuinely earns its keep, and raising our
    # share toward 1.0 walks back down to our-arm-alone. Only 0.70 stays, as one bracketing read.
    # The two-group candidate was the DOCUMENTED failure mode: the public fork runs exactly that
    # per-target grid search on gold-58, and nested validation measured it at +0.011 in-sample vs
    # +0.0003 out-of-fold. Not spending a slot on a known overfit.
    #
    # ⭐ Centre-concentrated windows get four of the five slots because it is the only lever with an
    # independent, fully specified, measured ladder (a published result, topic 737597):
    #     192px / no crop / 24 spread      0.7695 OOF
    #     288px / 140mm crop / 12 adjacent 0.7777
    #     288px / 140mm crop / centre-9    0.7815
    # "Slice geometry mattered more than resolution." Our 20 members average ALL 10 sliding
    # positions across the band - the losing arm of that exact comparison. 0.940 is the K=10 point,
    # so these four map the whole curve in one day.
    {
        "name": "centre 8 windows",
        "dir": os.path.join(HERE, "variants", "push_center8"),
        "slug": "knee-mri-centre-8-windows",
        "banked": 20,
        "expect": ["centre-concentrated windows:"],
        "msg": ("Each model reads the knee at the middle eight of its ten slice positions instead of all "
                "ten. The findings we score sit near the centre of the stack, so the outer positions "
                "mostly add background. Models and blend weight unchanged."),
    },
    {
        "name": "FAST: partner 4 + arm 48 centred (2.3x less work)",
        "dir": os.path.join(HERE, "variants", "push_fast"),
        "slug": "knee-mri-fast-centre-both-halves",
        "banked": 20,
        "expect": ["FAST: partner windows ->"],
        "msg": ("Both halves now read far fewer positions, taken from the middle of the range rather "
                "than spread across all of it. The partner models read four positions instead of ten "
                "and our own model reads forty eight instead of ninety four, drawn from the middle of "
                "each of the five image series. That is roughly a third of the work of the current "
                "entry. Our own earlier reading showed most of that sampling was redundant."),
    },
    {
        "name": "centre 4 windows",
        "dir": os.path.join(HERE, "variants", "push_center4"),
        "slug": "knee-mri-centre-4-windows",
        "banked": 20,
        "expect": ["centre-concentrated windows:"],
        "msg": ("Each model reads the knee at the middle four of its ten slice positions instead of all "
                "ten. The findings we score sit near the centre of the stack, so the outer positions "
                "mostly add background. Models and blend weight unchanged."),
    },
    {
        "name": "arm per-slot centre, 48 windows (faster AND targeted)",
        "dir": os.path.join(HERE, "variants", "push_armcentre48"),
        "slug": "knee-mri-arm-per-slot-centre-48",
        "banked": 20,
        "expect": ["ARM per-slot centre: kept"],
        "msg": ("Our own model now reads forty eight positions instead of ninety four, and takes them "
                "from the middle of each of the five image series rather than spreading them across "
                "the whole stack. Our own earlier reading showed forty two positions scored almost as "
                "well as sixty two, so most of that sampling was redundant. This is faster and reads "
                "the part of each series where the findings actually sit."),
    },
    {
        "name": "our half weight 0.70",
        "dir": os.path.join(HERE, "variants", "push_armw070"),
        "slug": "knee-mri-our-half-weight-070",
        "banked": 20,
        "expect": ["[V18] CoAtNet target weights: ACL=0.70"],
        "msg": ("Our own model carries seventy percent of the prediction instead of fifty five, with "
                "the partner models keeping the rest. Nothing else changes."),
    },
]


def log(m):
    line = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} UTC | {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(args, timeout=900):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as exc:
        return f"__EXC__ {type(exc).__name__}: {exc}"


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


def kernel_text(slug):
    """Full stdout of a finished kernel run, or None."""
    out_dir = os.path.join(HERE, "runlogs", slug)
    os.makedirs(out_dir, exist_ok=True)
    sh(["kaggle", "kernels", "output", "dreaddevelopment/" + slug, "-p", out_dir], timeout=1200)
    path = os.path.join(out_dir, slug + ".log")
    if not os.path.exists(path):
        return None
    try:
        entries = json.load(open(path, encoding="utf-8"))
        return "\n".join(str(e.get("data", "")) for e in entries)
    except Exception:
        return None


def verify(cand, txt):
    """Health-check the run against its own log before a submission slot is spent."""
    banked = [l for l in txt.split("\n") if "banked" in l]
    pkg = next((l for l in txt.split("\n") if "weights package:" in l), "")
    arm = next((l for l in txt.split("\n") if "[arm 0] loaded" in l), "")
    checks = [
        ("no exception", "Traceback (most recent call last)" not in txt),
        ("our arm loaded", ARM in arm),
        ("weights package read", "20 member(s)" in pkg),
        (f"{cand['banked']} members banked", len(banked) == cand["banked"]),
        ("V18 calibration applied", "[V18] 88-feature transformer calibration applied" in txt),
        ("final blend fired", "V18 calibrated transformer + CoAtNet" in txt),
        ("fallback NOT taken", "CoAtNet output unavailable" not in txt),
    ]
    for e in cand["expect"]:
        checks.append((f"change took effect: {e!r}", e in txt))
    for name, ok in checks:
        log(f"    [{'OK ' if ok else 'BAD'}] {name}")
    marker = next((l.strip() for l in txt.split("\n") if cand["expect"][0] in l), None)
    if marker:
        log(f"    marker: {marker[:150]}")
    return all(ok for _, ok in checks)


def slots_used_today():
    """How many of the 5 daily submission slots are already spent (UTC day)."""
    out = sh(["kaggle", "competitions", "submissions", "-c", COMP, "-v"], timeout=300)
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for l in out.splitlines() if today in l)


def wait_for_slot():
    """Block until a daily submission slot is free. A verified run is worth waiting for;
    submitting into a full day just burns the candidate and reads nothing."""
    while True:
        n = slots_used_today()
        if n < 5:
            log(f"  submission slots: {n}/5 used today -> slot available")
            return
        log(f"  submission slots: {n}/5 used today -> holding for the 00:00 UTC reset")
        time.sleep(900)


def submit(cand):
    for v in ("1", "2", "3"):
        out = sh(["kaggle", "competitions", "submit", "-c", COMP,
                  "-k", "dreaddevelopment/" + cand["slug"], "-v", v,
                  "-f", "submission.csv", "-m", cand["msg"]])
        low = out.lower()
        if "successfully submitted" in low or "submissions remaining" in low:
            log(f"  SUBMITTED '{cand['name']}' (version {v}): {out[:160]}")
            return True
        if "version" in low:
            continue
        log(f"  SUBMIT FAILED '{cand['name']}': {out[:300]}")
        return False
    log(f"  SUBMIT FAILED '{cand['name']}': no usable version")
    return False


def status_of(slug):
    return sh(["kaggle", "kernels", "status", "dreaddevelopment/" + slug], timeout=300).lower()


def main():
    """Push and verify EVERYTHING first; submit as slots free.

    The previous version called wait_for_slot() the moment a candidate verified, which blocked the
    single-threaded loop - so with the day's five submissions already spent, one verified candidate
    stalled the other four from even being pushed. Verification is free and submission is the scarce
    resource, so they are now decoupled: candidates are banked as soon as they verify, and banked
    candidates are submitted whenever a slot is available. At the reset the whole bank fires at once
    instead of trickling out one 16-minute kernel run at a time.
    """
    pending = list(CANDIDATES)
    inflight, ready, finished = [], [], []
    log(f"campaign armed: {len(pending)} candidate(s) - verify-first, submit-when-free")

    while pending or inflight or ready:
        # --- submit anything already banked, if the day has room ---
        while ready:
            try:
                used = slots_used_today()
            except Exception as exc:
                log(f"slot read failed ({type(exc).__name__}: {exc})"); break
            if used >= 5:
                log(f"  {len(ready)} verified and banked; slots {used}/5 used - holding")
                break
            cand = ready.pop(0)
            finished.append((cand, "submitted" if submit(cand) else "submit-failed"))

        # --- start work if there is room and allowance ---
        while pending and len(inflight) < MAX_INFLIGHT:
            q = quota()
            if q is None:
                time.sleep(120); break
            used_h, allowed, frozen = q
            if used_h > allowed - RESERVE_H:
                log(f"blocked: {used_h:.1f}/{allowed:.1f} h used"); time.sleep(300); break
            cand = pending[0]
            log(f"quota clear ({used_h:.1f}/{allowed:.1f} h) -> pushing '{cand['name']}'")
            out = sh(["kaggle", "kernels", "push", "-p", cand["dir"]], timeout=1800)
            if "successfully pushed" in out.lower():
                log(f"  pushed: {out[:160]}")
                pending.pop(0); inflight.append(cand); time.sleep(60)
            else:
                log(f"  push failed ('{cand['name']}'): {out[:200]} -- retry shortly")
                time.sleep(150); break

        if not inflight:
            time.sleep(120)
            continue

        # --- watch what is running ---
        time.sleep(150)
        for cand in list(inflight):
            st = status_of(cand["slug"])
            if "complete" in st:
                log(f"run COMPLETE: '{cand['name']}' -- verifying")
                txt = kernel_text(cand["slug"])
                if txt is None:
                    log("  could not read the run log; leaving it for manual review")
                    inflight.remove(cand); finished.append((cand, "no-log")); continue
                if verify(cand, txt):
                    log(f"  VERDICT: ready -> BANKED '{cand['name']}' ({len(ready)+1} waiting to submit)")
                    ready.append(cand)
                else:
                    log(f"  VERDICT: DO NOT SUBMIT '{cand['name']}' -- slot preserved")
                    finished.append((cand, "rejected"))
                inflight.remove(cand)
            elif "error" in st or "cancel" in st:
                log(f"run FAILED: '{cand['name']}': {st[:160]}")
                inflight.remove(cand); finished.append((cand, "run-failed"))

    log("campaign done: " + ", ".join(f"{c['name']}={s}" for c, s in finished))


if __name__ == "__main__":
    sys.exit(main())
