#!/bin/bash
# Pull a trained arm back and PROVE it arrived intact before anything is destroyed.
#
# The previous version chunked with integer-megabyte dd offsets, which do not tile a file whose
# size is not a multiple of a megabyte: the pieces overlapped, and truncating to the right length
# with head -c made the size check pass on corrupt bytes. A size check that can be satisfied by
# truncation proves nothing. This uses one stream and compares SHA256 end to end.
set -u
K=/c/Users/johna/.ssh/ddp_ed25519; H="$1"; P="$2"; SRC="$3"; DST="$4"
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no"
REM=$($SSH root@$H "sha256sum $SRC | cut -d' ' -f1; stat -c '%s' $SRC" 2>/dev/null | tr -d '\r')
RHASH=$(echo "$REM" | head -1); RSIZE=$(echo "$REM" | tail -1)
echo "remote sha256 $RHASH  size $RSIZE"
$SSH root@$H "cat $SRC" > "$DST"
LSIZE=$(stat -c '%s' "$DST"); LHASH=$(sha256sum "$DST" | cut -d' ' -f1)
echo "local  sha256 $LHASH  size $LSIZE"
if [ "$RHASH" = "$LHASH" ] && [ "$RSIZE" = "$LSIZE" ]; then echo "VERIFIED OK"; exit 0; else echo "MISMATCH - DO NOT DESTROY THE BOX"; exit 1; fi
