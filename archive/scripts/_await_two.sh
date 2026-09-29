#!/bin/bash
# Bank both overnight arms the moment each passes its likely peak, md5-verified at both ends.
# Every previous arm peaked at ep14 of 24 and then rolled over, so pull from ep15 onward and keep
# re-pulling while the best still improves. Losing a finished arm to a teardown has happened before.
K=/c/Users/johna/.ssh/id_ed25519
B1="ssh -i $K -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 20566 root@192.220.55.116"
B2="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p 58677 root@38.49.42.46"
LOG=D:/claude2/main_track/knee/_await_two.log
mkdir -p ship_parity ship_scanner24
pull () {  # $1=ssh  $2=port  $3=host  $4=tag  $5=outdir
  RH=$($1 "md5sum /workspace/knee/raptor_ft_$4.pt" 2>/dev/null | awk '{print $1}')
  [ -z "$RH" ] && return 1
  scp -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $2 \
      root@$3:/workspace/knee/raptor_ft_$4.pt D:/claude2/main_track/knee/$5/raptor_ft_$4.pt 2>/dev/null
  LH=$(md5sum D:/claude2/main_track/knee/$5/raptor_ft_$4.pt 2>/dev/null | awk '{print $1}')
  echo "$(date -u +%H:%M) $4 remote=$RH local=$LH match=$([ "$RH" = "$LH" ] && echo YES || echo NO)" >> $LOG
}
for i in $(seq 1 300); do
  N1=$($B1 'grep -c "^ep" /workspace/knee/run_PARITY_S5.log 2>/dev/null' 2>/dev/null | tr -d '\r')
  N2=$($B2 'grep -c "^ep" /workspace/knee/run_SCANNER_E24.log 2>/dev/null' 2>/dev/null | tr -d '\r')
  E1=$($B1 'grep "^ep" /workspace/knee/run_PARITY_S5.log 2>/dev/null | tail -1' 2>/dev/null)
  E2=$($B2 'grep "^ep" /workspace/knee/run_SCANNER_E24.log 2>/dev/null | tail -1' 2>/dev/null)
  echo "$(date -u +%H:%M) PARITY n=$N1 | $E1" >> $LOG
  echo "$(date -u +%H:%M) SCAN24 n=$N2 | $E2" >> $LOG
  [ "${N1:-0}" -ge 16 ] && pull "$B1" 20566 192.220.55.116 PARITY_S5   ship_parity
  [ "${N2:-0}" -ge 16 ] && pull "$B2" 58677 38.49.42.46    SCANNER_E24 ship_scanner24
  if [ "${N1:-0}" -ge 24 ] && [ "${N2:-0}" -ge 24 ]; then echo "$(date -u +%H:%M) both complete" >> $LOG; break; fi
  sleep 600
done
