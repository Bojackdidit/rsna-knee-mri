#!/usr/bin/env python3
"""Parallel range-request downloader for the Kaggle competition bundle.

Kaggle's single stream caps around 25-30 MB/s regardless of host bandwidth. The API redirects to a
Google Cloud Storage signed URL, and GCS honours HTTP Range, so N concurrent range requests scale
well past the single-stream cap. This is the whole reason the download looked slow.
"""
import os, sys, threading, time, urllib.request, json

COMP = "rsna-knee-abnormality-detection"
OUT = "/workspace/%s.zip" % COMP
N = int(os.environ.get("NCONN", "24"))
tok = open("/root/.kaggle/access_token").read().strip()
try:
    tok = json.loads(tok).get("access_token", tok)
except Exception:
    pass

api = "https://www.kaggle.com/api/v1/competitions/data/download-all/" + COMP
req = urllib.request.Request(api, headers={"Authorization": "Bearer " + tok})
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
op = urllib.request.build_opener(NoRedirect)
try:
    op.open(req)
    sys.exit("expected a redirect")
except urllib.error.HTTPError as e:
    if e.code not in (301, 302, 303, 307): raise
    url = e.headers["Location"]
print("signed url ok", flush=True)

with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
    total = int(r.headers["Content-Length"])
print("size %.1f GB | %d connections" % (total / 1e9, N), flush=True)

with open(OUT, "wb") as f:
    f.truncate(total)
step = (total + N - 1) // N
done = [0] * N
lock = threading.Lock()

def worker(i):
    s = i * step
    e = min(s + step, total) - 1
    if s > e: return
    pos = s
    while pos <= e:
        try:
            rq = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (pos, e)})
            with urllib.request.urlopen(rq, timeout=120) as r, open(OUT, "r+b") as f:
                f.seek(pos)
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); pos += len(b)
                    with lock: done[i] += len(b)
        except Exception as ex:
            print("  conn %d retry at %d (%s)" % (i, pos, type(ex).__name__), flush=True)
            time.sleep(2)

ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
t0 = time.time()
for t in ts: t.start()
last = 0
while any(t.is_alive() for t in ts):
    time.sleep(20)
    with lock: cur = sum(done)
    el = time.time() - t0
    print("  %.1f/%.1f GB | %.0f MB/s inst | %.0f MB/s avg | eta %.0f min"
          % (cur / 1e9, total / 1e9, (cur - last) / 20 / 1e6, cur / el / 1e6,
             (total - cur) / max(1, cur / el) / 60), flush=True)
    last = cur
for t in ts: t.join()
print("DONE %.1f GB in %.0f min" % (total / 1e9, (time.time() - t0) / 60), flush=True)
