#!/usr/bin/env python3
"""Evidence-gated RSNA submits for the v11 arm (Dread 2026-08-26: only gains, no guesses).

Chain, each step gated on the previous:
  1. wait for fire_queue to PUSH the w0.50 kernel, then for its run to COMPLETE
  2. verify_run.py must say READY TO SUBMIT (arm loaded, 20 members banked, no fallback)
  3. submit w0.50 (version 1 - fresh slug, confirmed 404 before push)
  4. wait for the board score of that submit
  5. w0.55 is submitted ONLY if w0.50 scored ABOVE the incumbent 0.938 - i.e. the v11 arm
     is a confirmed board gain, making the weight read on it evidence, not a guess.
     Otherwise HOLD and stop.
Every decision is printed; a watcher notifies the session. A failed submit costs no slot.
"""
import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BV = os.path.join(HERE, "phase2", "blend_variants")
COMP = "rsna-knee-abnormality-detection"
INCUMBENT = 0.938
ARM = "raptor_ft_coatnet_v11blend_full_swa"
K50 = "dreaddevelopment/knee-mri-fine96-blend-cross-team"
K55 = "dreaddevelopment/knee-mri-fine96-blend-w55"
MSG50 = ("Cross team blend with our newest CoAtNet arm swapped in. The arm was retrained on the "
         "finer 96 slice corpus with the blended labels and is the strongest we have measured "
         "offline. The blend weight stays at the half and half setting the board confirmed.")
MSG55 = ("Same blend but our arm carries 55 percent of the weight instead of 50. The weight was "
         "tuned against a weaker arm, so the stronger arm may deserve more of it.")

def sh(args, cwd=None):
    r = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return (r.stdout or "") + (r.stderr or "")

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

def wait_pushed(tag, hours=14):
    end = time.time() + hours * 3600
    path = os.path.join(HERE, "fire_queue_out.log")
    while time.time() < end:
        txt = open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""
        if ("PUSHED  %s" % tag) in txt:
            log("push confirmed: %s" % tag)
            return True
        time.sleep(120)
    log("TIMEOUT waiting for push of %s" % tag)
    return False

def wait_complete(slug, hours=8):
    end = time.time() + hours * 3600
    while time.time() < end:
        out = sh(["kaggle", "kernels", "status", slug]).lower()
        if "complete" in out:
            log("kernel COMPLETE: %s" % slug)
            return True
        if "error" in out or "cancel" in out:
            log("KERNEL FAILED: %s -> %s" % (slug, out.strip()[:200]))
            return False
        time.sleep(180)
    log("TIMEOUT waiting for kernel %s" % slug)
    return False

def verified(slug):
    out = sh([sys.executable, os.path.join(BV, "verify_run.py"), slug.split("/")[1], ARM], cwd=BV)
    print(out, flush=True)
    return "READY TO SUBMIT" in out

def submit(slug, msg):
    for v in ("1", "2"):
        out = sh(["kaggle", "competitions", "submit", "-c", COMP, "-k", slug, "-v", v,
                  "-f", "submission.csv", "-m", msg])
        low = out.lower()
        if ("successfully submitted" in low) or ("submissions remaining" in low):
            log("SUBMITTED %s (version %s): %s" % (slug, v, out.strip()[:160]))
            return True
        if "version" in low:
            log("version %s rejected for %s, trying next" % (v, slug))
            continue
        log("SUBMIT FAILED %s: %s" % (slug, out.strip()[:300]))
        return False
    return False

def sub_rows():
    import csv, io
    out = sh(["kaggle", "competitions", "submissions", "-c", COMP, "-v"])
    try:
        return list(csv.DictReader(io.StringIO(out)))
    except Exception:
        return []

def newest_ref():
    rows = sub_rows()
    return rows[0].get("ref") if rows else None

def newest_score(prev_ref, hours=8):
    """Wait for a NEW submission row (ref != prev_ref) to finish scoring; return its score."""
    end = time.time() + hours * 3600
    while time.time() < end:
        rows = sub_rows()
        if rows and rows[0].get("ref") != prev_ref:
            row = rows[0]  # newest first
            status = (row.get("status") or "").lower()
            if "error" in status:
                log("SUBMISSION ERRORED: %s" % row)
                return None
            if "complete" in status:
                try:
                    return float(row.get("publicScore") or row.get("publicScoreNullable"))
                except (TypeError, ValueError):
                    log("complete but no numeric score: %s" % row)
                    return None
        time.sleep(600)
    log("TIMEOUT waiting for score")
    return None

def push(path):
    out = sh(["kaggle", "kernels", "push", "-p", path])
    ok = "successfully pushed" in out.lower()
    if ok:
        log("PUSHED %s" % path)
    else:
        log("PUSH FAILED %s: %s" % (path, out.strip()[:200]))
    return ok

def main():
    # w0.50 was already pushed by hand (version 1, running). Exactly 7 GPU hours exist:
    # the two v11 runs and NOTHING else. w0.55's run starts only after w0.50's run
    # completes AND verifies, so a systemic kernel fault cannot burn both runs.
    if not wait_complete(K50):
        log("HOLD: w0.50 run did not complete - nothing submitted, w0.55 NOT pushed.")
        return 1
    if not verified(K50):
        log("HOLD: w0.50 run failed verification - NOT submitting, w0.55 NOT pushed. "
            "Human review needed.")
        return 1
    ref0 = newest_ref()
    if not submit(K50, MSG50):
        return 1
    # its scoring runs on Kaggle's side; start the second run in parallel with that
    if not push(os.path.join(BV, "push_v11_w55")):
        log("HOLD: w0.55 push failed - fix by hand.")
        return 1
    if not wait_complete(K55):
        log("NOTE: w0.55 run did not complete; its submit is off the table.")
    s = newest_score(ref0)
    if s is None:
        log("HOLD: no score for w0.50 - w0.55 NOT submitted.")
        return 1
    log("BOARD SCORE w0.50 = %.5f (incumbent %.3f)" % (s, INCUMBENT))
    if s <= INCUMBENT:
        log("HOLD: w0.50 did not beat %.3f - v11 arm is not a confirmed board gain; "
            "w0.55 stays unsubmitted per Dread's order." % INCUMBENT)
        return 0
    log("GAIN CONFIRMED (%.5f > %.3f) - proceeding to the w0.55 weight read." % (s, INCUMBENT))
    if not verified(K55):
        log("HOLD: w0.55 run failed verification - NOT submitting.")
        return 1
    ref1 = newest_ref()
    if not submit(K55, MSG55):
        return 1
    s2 = newest_score(ref1)
    if s2 is not None:
        log("BOARD SCORE w0.55 = %.5f" % s2)
    log("chain complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
