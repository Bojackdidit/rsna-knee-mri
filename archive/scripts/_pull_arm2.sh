#!/bin/bash
# Bank arm 2's best-so-far checkpoint every 25 minutes. Credit runs out before the run finishes, so
# the box can vanish mid-epoch; the trainer writes best-so-far on every improvement, and this makes
# sure whatever it last wrote is on disk here rather than on a machine we no longer have.
set -u
H=192.220.55.116; P=20566
K=/c/Users/johna/.ssh/id_ed25519
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
mkdir -p arms_v13
for i in $(seq 1 40); do
  ep=$($SSH 'cd /workspace/knee; grep -c "^ep" run_SCRATCH_AUG2.log' 2>/dev/null)
  rs=$($SSH 'stat -c %s /workspace/knee/raptor_ft_SCRATCH_AUG2.pt 2>/dev/null' 2>/dev/null)
  if [ -z "${rs:-}" ]; then echo "$(date -u +%H:%M:%S) box unreachable - stopping"; exit 3; fi
  scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      root@$H:/workspace/knee/raptor_ft_SCRATCH_AUG2.pt arms_v13/_tmp.pt 2>/dev/null
  ls=$(stat -c %s arms_v13/_tmp.pt 2>/dev/null)
  if [ "${ls:-0}" = "${rs:-1}" ]; then
    mv -f arms_v13/_tmp.pt arms_v13/raptor_ft_SCRATCH_AUG2.pt
    echo "$(date -u +%H:%M:%S) banked at ~ep${ep:-?}  $ls bytes  (sizes match)"
  else
    echo "$(date -u +%H:%M:%S) SIZE MISMATCH local=$ls remote=$rs - kept previous"
    rm -f arms_v13/_tmp.pt
  fi
  $SSH 'cd /workspace/knee; grep -c "SCRATCH_AUG2 done" ft_chain.log' 2>/dev/null | grep -q '^[1-9]' && { echo "run finished"; exit 0; }
  sleep 1500
done
