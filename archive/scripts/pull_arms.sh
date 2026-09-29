#!/bin/bash
# Pull each trained arm off the box THE MOMENT its run finishes.
#
# memory/pull-weights-before-teardown: "a gate npz without a .pt is an arm you cannot ship; the best
# upgrade was lost this way". A box can be stopped by credit running out, by the host, or by me.
#
# FIXED 2026-09-13 22:45: the old version grepped for "^A done" while chain_v11.sh writes
# "V11_AUG_A done" -- it would have waited 20 h and pulled nothing. Also now covers all FOUR folds.
set -u
cd "$(dirname "$0")"
K=/c/Users/johna/.ssh/id_ed25519
H=154.64.230.50
P=53473
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
mkdir -p arms_v11

pull_one () {
  tag="V11_AUG_$1"
  for i in $(seq 1 1500); do         # 25 h ceiling per arm at 60 s polling
    if $SSH "grep -q '^${tag} done' /workspace/knee/chain_v11.log 2>/dev/null" 2>/dev/null; then
      echo "$(date -u +%H:%M:%S) ${tag}: run finished, pulling"
      for f in "raptor_ft_${tag}.pt" "raptor_gold_${tag}.npz" "raptor_ft_${tag}.json" \
               "valids_${tag}.csv" "run_${tag}.log"; do
        $SCP "root@$H:/workspace/knee/$f" arms_v11/ 2>/dev/null
      done
      # hash both ends -- memory/verify-artifacts-with-hashes: a size check proves nothing
      R=$($SSH "md5sum /workspace/knee/raptor_ft_${tag}.pt 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
      L=$(md5sum "arms_v11/raptor_ft_${tag}.pt" 2>/dev/null | cut -d' ' -f1)
      if [ -n "$R" ] && [ "$R" = "$L" ]; then echo "  md5 MATCH $L"; else echo "  MD5 MISMATCH box=$R local=$L"; fi
      python - "$tag" <<'PYEOF'
import sys, json, os
t = sys.argv[1]
p = "arms_v11/raptor_ft_%s.json" % t
if os.path.exists(p):
    d = json.load(open(p))
    print("  %s best_gold_auc %.4f over %d epochs, n_train %s"
          % (t, d.get("best_gold_auc", 0), len(d.get("hist", [])), d.get("n_train")))
    a = d.get("aucs", {})
    if a:
        worst = sorted(a.items(), key=lambda kv: kv[1])[:4]
        print("  weakest: " + ", ".join("%s %.3f" % (k, v) for k, v in worst))
else:
    print("  WARNING no json for %s -- run may have died" % t)
PYEOF
      return 0
    fi
    sleep 60
  done
  echo "$(date -u +%H:%M:%S) ${tag}: timed out waiting"
  return 1
}

for t in A B C D; do pull_one "$t"; done
echo "=== all arms pulled $(date -u +%H:%M:%S) ==="
ls -la arms_v11/ | awk '{print $5,$9}'
