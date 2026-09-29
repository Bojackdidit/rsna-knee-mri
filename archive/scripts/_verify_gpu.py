#!/usr/bin/env python3
"""Verify every GPU holds full clocks UNDER SUSTAINED LOAD before committing a run to it.

Idle clocks always look fine - the previous host read 1980 MHz idle and collapsed to 630 MHz at
87C once training started, which is how half a training slot got wasted. This puts a real load on
each card for 90s and fails the box if any card droops or reports a thermal slowdown.
"""
import subprocess, sys, time
import torch

FAIL = []
n = torch.cuda.device_count()
print("GPUs visible: %d" % n, flush=True)
if n == 0:
    sys.exit("FAIL: no CUDA devices")


def smi(q):
    return subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()


maxclk = [float(x) for x in smi("clocks.max.sm")]
name = smi("name")
streams = []
for d in range(n):
    torch.cuda.set_device(d)
    a = torch.randn(8192, 8192, device="cuda:%d" % d, dtype=torch.bfloat16)
    streams.append(a)

print("loading all cards for 90s...", flush=True)
t0 = time.time()
samples = {d: [] for d in range(n)}
while time.time() - t0 < 90:
    for d in range(n):
        torch.cuda.set_device(d)
        a = streams[d]
        for _ in range(6):
            a = a @ streams[d]
    torch.cuda.synchronize()
    clk = [float(x) for x in smi("clocks.sm")]
    tmp = [float(x) for x in smi("temperature.gpu")]
    for d in range(n):
        samples[d].append((clk[d], tmp[d]))

thr = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True).stdout
slow = thr.count("SW Thermal Slowdown           : Active") + thr.count("HW Thermal Slowdown           : Active")

print("\n%-4s %-16s %-9s %-9s %-8s %s" % ("gpu", "name", "clk_med", "clk_max", "temp", "verdict"))
for d in range(n):
    cl = sorted(s[0] for s in samples[d])
    med = cl[len(cl) // 2]
    tp = max(s[1] for s in samples[d])
    frac = med / maxclk[d]
    ok = frac >= 0.80 and tp < 85
    if not ok:
        FAIL.append("gpu%d clk %.0f/%.0f (%.0f%%) temp %.0f" % (d, med, maxclk[d], 100 * frac, tp))
    print("%-4d %-16s %-9.0f %-9.0f %-8.0f %s"
          % (d, name[d][:16], med, maxclk[d], tp, "OK" if ok else "**DEGRADED**"))
if slow:
    FAIL.append("thermal slowdown active on %d card(s)" % slow)
    print("thermal slowdown reported active")

if FAIL:
    print("\nVERDICT: REJECT THIS BOX -> " + "; ".join(FAIL), flush=True)
    sys.exit(2)
print("\nVERDICT: ALL CARDS GOOD - safe to start the pipeline", flush=True)
