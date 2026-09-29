#!/usr/bin/env python3
"""Restore the v11 96-slice corpus from dreaddevelopment/knee-corpus-v11-stacks.

The local all_vols.npy lived in a session temp scratchpad and was deleted by the cleanup that
followed the 2026-09-25 power outage. The Kaggle dataset holds the same bytes split into five
.part files (the split exists because a single ~48 GB object cannot be pulled off Kaggle in one
stream). This script downloads part by part, streams each zip straight into the growing
all_vols.npy (no extract-then-concat double storage), deletes the zip, and verifies the result.

Resumable: _restore_state.json records how many parts are fully appended; on restart the output
is truncated back to the last good byte count and the next part continues.
"""
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

DS = "dreaddevelopment/knee-corpus-v11-stacks"
DST = r"D:\claude2\main_track\knee\corpus_v11"
PARTS = [("all_vols.00.part", 11811160064), ("all_vols.01.part", 11811160064),
         ("all_vols.02.part", 11811160064), ("all_vols.03.part", 11811160064),
         ("all_vols.04.part", 518496384)]
TOTAL = sum(s for _, s in PARTS)          # 47,763,136,640 = 128B npy header + 4407*96*336*336
SHAPE = (4407, 96, 336, 336)
OUT = os.path.join(DST, "all_vols.npy")
STATE = os.path.join(DST, "_restore_state.json")


AUTH_CFG = os.path.join(DST, "_curl_auth.cfg")


def write_auth_cfg():
    # local auth is the CLI's OAuth access token, not the classic username/key kaggle.json
    tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
    with open(AUTH_CFG, "w") as fh:
        fh.write('header = "Authorization: Bearer %s"\n' % tok)


def fetch(name, expect_size):
    """Download one dataset file with retries; return path to the downloaded artifact."""
    for attempt in range(1, 4):
        r = subprocess.run(["kaggle", "datasets", "download", DS, "-f", name, "-p", DST, "--force"],
                           capture_output=True, text=True, timeout=4 * 3600)
        zp, raw = os.path.join(DST, name + ".zip"), os.path.join(DST, name)
        if os.path.exists(zp):
            return zp
        if os.path.exists(raw) and (os.path.getsize(raw) == expect_size
                                    or (expect_size < 0 and os.path.getsize(raw) > 0)):
            return raw
        print("  attempt %d failed: %s" % (attempt, (r.stderr or r.stdout).strip()[-300:]), flush=True)
        time.sleep(30)
    raise RuntimeError("download failed 3x: " + name)


def fetch_part(name, expect_size):
    """curl with byte-range resume. The kaggle CLI hung at 0 bytes on the 11.8 GB parts
    (2026-09-25); curl with --speed-limit aborts a stall in 60 s and -C - resumes cleanly."""
    out = os.path.join(DST, name + ".dl")
    url = "https://www.kaggle.com/api/v1/datasets/download/%s/%s" % (DS, name)
    for attempt in range(1, 25):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have == expect_size:
            return out
        # junk guard: an error body or a range-ignored restart poisons the offset - start clean
        if have > expect_size or (0 < have < 1_000_000 and attempt > 1):
            os.remove(out)
        r = subprocess.run(["curl", "-sS", "-L", "-K", AUTH_CFG, "-C", "-",
                            "--connect-timeout", "30",
                            "--speed-limit", "10240", "--speed-time", "60",
                            "-o", out, url],
                           capture_output=True, text=True, timeout=3 * 3600)
        have = os.path.getsize(out) if os.path.exists(out) else 0
        print("  curl attempt %d rc=%d have %.2f/%.2f GB %s"
              % (attempt, r.returncode, have / 1e9, expect_size / 1e9,
                 (r.stderr or "").strip()[-200:]), flush=True)
        if have == expect_size:
            return out
        if r.returncode == 33:      # server ignored the Range header - resume impossible
            if os.path.exists(out):
                os.remove(out)
        time.sleep(15)
    raise RuntimeError("curl failed 24x: " + name)


def append_stream(src, out_fh, expect_size):
    """Stream src (zip member or raw file) onto out_fh; return bytes written."""
    n = 0
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            assert len(names) == 1, names
            info = z.getinfo(names[0])
            assert info.file_size == expect_size, (info.file_size, expect_size)
            with z.open(names[0]) as fh:
                while True:
                    buf = fh.read(64 << 20)
                    if not buf:
                        break
                    out_fh.write(buf)
                    n += len(buf)
    else:
        with open(src, "rb") as fh:
            while True:
                buf = fh.read(64 << 20)
                if not buf:
                    break
                out_fh.write(buf)
                n += len(buf)
    return n


def main():
    os.makedirs(DST, exist_ok=True)
    t0 = time.time()

    for small in ("all_ids.npy", "all_masks.npy"):
        if not os.path.exists(os.path.join(DST, small)):
            got = fetch(small, -1)
            if got.endswith(".zip"):
                with zipfile.ZipFile(got) as z:
                    z.extractall(DST)
                os.remove(got)
            print("got %s" % small, flush=True)

    done = 0
    if os.path.exists(STATE):
        done = json.load(open(STATE))["parts_done"]
    good = sum(s for _, s in PARTS[:done])
    if os.path.exists(OUT) and os.path.getsize(OUT) != good:
        print("truncating %s from %d back to last good %d" % (OUT, os.path.getsize(OUT), good), flush=True)
        with open(OUT, "r+b") as fh:
            fh.truncate(good)
    elif not os.path.exists(OUT):
        open(OUT, "wb").close()

    write_auth_cfg()
    for i in range(done, len(PARTS)):
        name, size = PARTS[i]
        print("=== part %d/5 %s (%.1f GB) el=%.0fm ===" % (i + 1, name, size / 1e9,
                                                           (time.time() - t0) / 60), flush=True)
        src = fetch_part(name, size)
        with open(OUT, "ab") as out_fh:
            n = append_stream(src, out_fh, size)
        assert n == size, (n, size)
        os.remove(src)
        json.dump({"parts_done": i + 1}, open(STATE, "w"))
        print("  appended, total now %.1f/%.1f GB" % (os.path.getsize(OUT) / 1e9, TOTAL / 1e9), flush=True)

    assert os.path.getsize(OUT) == TOTAL, os.path.getsize(OUT)
    vols = np.load(OUT, mmap_mode="r")
    assert vols.shape == SHAPE and vols.dtype == np.uint8, (vols.shape, vols.dtype)
    ids = np.load(os.path.join(DST, "all_ids.npy"), allow_pickle=True)
    masks = np.load(os.path.join(DST, "all_masks.npy"))
    assert len(ids) == SHAPE[0] and masks.shape == (SHAPE[0], SHAPE[1]), (len(ids), masks.shape)
    rng = np.random.default_rng(0)
    for i in rng.choice(SHAPE[0], 6, replace=False):
        derived = (vols[i].reshape(SHAPE[1], -1).sum(1) > 0).astype(np.uint8)
        assert (derived == masks[i]).all(), "mask mismatch study %d" % i
    if os.path.exists(AUTH_CFG):
        os.remove(AUTH_CFG)
    print("VERIFY PASS shape=%s ids=%d masks ok (6 spot checks) in %.0fm"
          % (vols.shape, len(ids), (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
