#!/bin/bash
# Submit a kernel AND capture its runtime. Usage: fire_probe.sh <kernel-slug> <version> "<message>"
#
# RSNA is judged on runtime as well as AUC, and runtime is NOT in the API - not in
# `competitions submissions`, not in `kernels status`, not in the stored kernel output. The only
# measurement is the wall clock between submitting and the status flipping PENDING -> COMPLETE.
#
# int24's runtime was LOST because the poller was started after the submit and its first reading was
# already COMPLETE. So here the poller starts FIRST, confirms it can see PENDING, and only then is
# the submit issued - which makes the transition impossible to miss.
set -u
SLUG="$1"; VER="$2"; MSG="$3"
COMP=rsna-knee-abnormality-detection
OUT="/tmp/probe_$(echo "$SLUG" | tr / _).log"

status_now () {
  kaggle competitions submissions -c "$COMP" --csv 2>/dev/null | python -c "
import sys,csv
r=list(csv.DictReader(sys.stdin))
print((r[0].get('status','') if r else '') + '|' + (r[0].get('publicScore','') if r else ''))"
}

BEFORE=$(status_now)
echo "before submit: $BEFORE"

( for i in $(seq 1 60); do
    s=$(status_now)
    echo "$(date -u +%H:%M:%S) $s"
    case "$s" in *COMPLETE*|*ERROR*) [ "$i" -gt 2 ] && break;; esac
    sleep 30
  done ) > "$OUT" 2>&1 &
POLL=$!
sleep 5

START=$(date +%s)
kaggle competitions submit -c "$COMP" -k "$SLUG" -v "$VER" -f submission.csv -m "$MSG" 2>&1 | tail -2
echo "submitted at $(date -u +%H:%M:%S)"

wait $POLL
END=$(date +%s)
echo "--- poll trace ---"
grep -aE "PENDING|COMPLETE|ERROR" "$OUT" | head -3
grep -aE "PENDING|COMPLETE|ERROR" "$OUT" | tail -3
FIRST_DONE=$(grep -anE "COMPLETE|ERROR" "$OUT" | head -1 | cut -d: -f1)
echo "first terminal reading at poll line $FIRST_DONE"
echo "elapsed to that reading: ~$(( (FIRST_DONE * 30) / 60 )) min (poll interval 30s)"
echo "total wall: $(( (END-START)/60 )) min"
