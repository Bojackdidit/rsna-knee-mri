#!/bin/bash
# Wait for SCANNER_ARM to reach ep14+, then pull the checkpoint and verify it by md5 at both ends.
# Both previous arms peaked at ep14, and credit runs dry ~04:40 UTC, so bank it early rather than
# betting on being awake for ep15. The trainer saves on BEST, so the file already holds the peak.
S="ssh -i /c/Users/johna/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 20566 root@192.220.55.116"
OUT=D:/claude2/main_track/knee/ship_scanner
LOG=D:/claude2/main_track/knee/_await_scanner.log
mkdir -p $OUT
for i in $(seq 1 400); do
  N=$($S 'grep -c "^ep" /workspace/knee/run_SCANNER_ARM.log' 2>/dev/null | tr -d "\r")
  BEST=$($S 'grep "^ep" /workspace/knee/run_SCANNER_ARM.log | tail -1' 2>/dev/null)
  echo "$(date -u +%H:%M) epochs=$N | $BEST" >> $LOG
  if [ "${N:-0}" -ge 15 ]; then
    echo "$(date -u +%H:%M) ep14 reached - pulling" >> $LOG
    RH=$($S 'md5sum /workspace/knee/raptor_ft_SCANNER_ARM.pt' 2>/dev/null | awk '{print $1}')
    scp -i /c/Users/johna/.ssh/id_ed25519 -o StrictHostKeyChecking=no -P 20566 \
        root@192.220.55.116:/workspace/knee/raptor_ft_SCANNER_ARM.pt $OUT/raptor_ft_SCANNER_ARM.pt 2>/dev/null
    LH=$(md5sum $OUT/raptor_ft_SCANNER_ARM.pt 2>/dev/null | awk '{print $1}')
    echo "$(date -u +%H:%M) remote=$RH local=$LH match=$([ "$RH" = "$LH" ] && echo YES || echo NO)" >> $LOG
    [ "$RH" = "$LH" ] && break
  fi
  sleep 300
done
echo "$(date -u +%H:%M) watcher done" >> $LOG
