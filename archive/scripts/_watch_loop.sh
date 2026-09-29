#!/bin/bash
# Active watcher for the RSNA box. Emits ONLY lines worth acting on.
#
# Two bugs fixed after they fired wrongly on the first night:
#  1. `pgrep -fc finetune_box.py` matched the watcher's OWN ssh command line (it contains that
#     string), so the count never reached 0 and the box sat idle for 13 minutes unreported.
#     `[f]inetune_box.py` cannot match itself.
#  2. the stall detector keyed on epoch counts, which stop advancing when training finishes and
#     SCORING starts - so it cried stall during a perfectly healthy gate_eval. It now only counts
#     a stall while a trainer is actually alive AND nothing else is running on the GPU.
cd /d/claude2/main_track/knee || exit 1
read IP PORT ID < /tmp/specbox
SEEN=""; IDLE_SENT=""; STALL_SENT=""; LASTEP=""; SAME=0; MISS=0; SSH_SENT=""
while true; do
  S=$(timeout 90 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$IP" \
      'cd /workspace/knee 2>/dev/null || exit 9
       for f in run_*.log; do
         [ -f "$f" ] || continue
         t=${f#run_}; t=${t%.log}
         B=$(grep -aoE "best [0-9]+\.[0-9]+" "$f" | tail -1 | sed "s/best //")
         [ -n "$B" ] && echo "BEST $t $B"
       done
       echo "EPSUM $(grep -acE "ep[0-9]+ loss" run_*.log 2>/dev/null | paste -sd, -)"
       echo "TRAIN $(pgrep -f "[f]inetune_(box|slice)\.py" 2>/dev/null | wc -l)"
       echo "OTHER $(pgrep -f "[g]ate_eval\.py" 2>/dev/null | wc -l)"
       echo "GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
       grep -ahoE "Traceback|CUDA error|out of memory|CUDA out of|Killed|AssertionError" run_*.log 2>/dev/null | sort -u | sed "s/^/ERR /"
       ls -1 raptor_ft_*.pt 2>/dev/null | sed "s/^/CKPT /"' 2>/dev/null)
  # vast.ai drops ssh briefly and its own banner says to retry - one failure is not an outage.
  # Alert only on two consecutive misses, so a hiccup does not read as a dead box overnight.
  if [ -z "$S" ]; then
    MISS=$((MISS+1))
    if [ "$MISS" -ge 2 ] && [ -z "$SSH_SENT" ]; then
      SSH_SENT=1
      echo "$(date -u +%H:%M:%S) *** BOX UNREACHABLE - 2 consecutive ssh failures ***"
    fi
    sleep 60; continue
  fi
  MISS=0; SSH_SENT=""
  EP=""; TR=0; OT=0; GU=0
  while IFS= read -r line; do
    case "$line" in
      BEST*|CKPT*|ERR*)
        case "$SEEN" in
          *"[$line]"*) ;;
          *) SEEN="$SEEN[$line]"; echo "$(date -u +%H:%M:%S) $line" ;;
        esac ;;
      EPSUM*) EP="$line" ;;
      TRAIN*) TR=${line#TRAIN } ;;
      OTHER*) OT=${line#OTHER } ;;
      GPU*)   GU=${line#GPU } ;;
    esac
  done <<< "$S"
  if [ "${TR:-0}" -eq 0 ] && [ "${OT:-0}" -eq 0 ]; then
    if [ -z "$IDLE_SENT" ]; then
      IDLE_SENT=1
      echo "$(date -u +%H:%M:%S) *** BOX IDLE - no trainer, no scorer, GPU ${GU}% - \$0.478/hr burning ***"
    fi
    SAME=0
  else
    IDLE_SENT=""
    if [ "${TR:-0}" -gt 0 ] && [ "${OT:-0}" -eq 0 ]; then
      if [ "$EP" = "$LASTEP" ]; then SAME=$((SAME+1)); else SAME=0; LASTEP="$EP"; STALL_SENT=""; fi
    else
      SAME=0
    fi
  fi
  # CoAtNet-384 epochs legitimately run past 20 minutes, so 15 was too tight and fired on a
# perfectly healthy run. 35 minutes without an epoch is a real problem; less is just slow.
# Threshold must sit ABOVE the slowest real epoch or it fires every epoch. CoAtNet-384 at
# 5 series / K=24 takes 35.9 min, so 35 was guaranteed to cry wolf. 55 gives real headroom.
if [ "$SAME" -ge 55 ] && [ -z "$STALL_SENT" ]; then
    STALL_SENT=1
    echo "$(date -u +%H:%M:%S) *** STALL - trainer alive, no epoch in 55+ min, GPU ${GU}% ***"
  fi
  sleep 60
done
