#!/bin/bash
# One-shot status for the spec chain. Everything I need per check, no nested quoting.
cd /workspace/knee 2>/dev/null || exit 1
# nvidia-smi -l produces nothing over a non-interactive ssh session, so sample with discrete calls
: > /tmp/u.txt
for i in $(seq 1 20); do
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >> /tmp/u.txt 2>/dev/null
  sleep 0.5
done
python3 - <<'PYEOF'
import os, re, time
v = []
try:
    for ln in open("/tmp/u.txt"):
        p = ln.split(",")
        if p and p[0].strip().isdigit():
            v.append(int(p[0]))
except Exception:
    pass
if v:
    s = sorted(v)
    print("GPU  mean %.0f%%  median %d%%  min %d%%  max %d%%  zeros %d/%d"
          % (sum(v)/len(v), s[len(s)//2], s[0], s[-1], v.count(0), len(v)))
else:
    print("GPU  no samples")
print("LOAD %s   cores %d" % (open("/proc/loadavg").read().split()[0:3], os.cpu_count()))
for tag in ("E0", "E1", "E1b", "E2", "E3b", "E4"):
    f = "run_%s.log" % tag
    if not os.path.exists(f):
        continue
    try:
        txt = open(f, errors="ignore").read()
    except Exception:
        continue
    eps = re.findall(r"ep(\d+) loss ([\d.]+) \| GOLD macro-AUC ([\d.]+) \(best ([\d.]+)\) \| (\d+)s", txt)
    tot = re.search(r"--epochs (\d+)", txt)
    val = re.search(r"validate (\d+)|validating on (\d+)", txt)
    age = (time.time() - os.path.getmtime(f)) / 60.0
    if eps:
        n, _, last, best, el = eps[-1]
        n = int(n) + 1
        per = int(el) / n
        want = int(tot.group(1)) if tot else 24
        eta = (want - n) * per / 60.0
        print("%-4s ep %2d/%-2d  last %.4f  best %s  %.0fs/ep  ETA %.0f min  (log %.1f min old)"
              % (tag, n, want, float(last), best, per, eta, age))
    else:
        err = [l for l in txt.splitlines() if "Error" in l or "Traceback" in l]
        print("%-4s no epochs yet  (log %.1f min old)%s" % (tag, age, "  ERR: " + err[-1][:60] if err else ""))
PYEOF
echo "CHAIN $(cat /workspace/chain.log 2>/dev/null | tr '\n' ' ')"
echo "TRAIN $(pgrep -fc 'finetune_box.py' 2>/dev/null) procs   DISK $(df -h /workspace | awk 'NR==2{print $4" free"}')"
ls -1 raptor_ft_*.pt 2>/dev/null | sed 's/^/CKPT  /'
