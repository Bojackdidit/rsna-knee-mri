#!/bin/bash
# Pull every arm off a box with the hash checked at BOTH ends. Usage: _collect_arms.sh PORT IP DESTDIR
#
# The rule exists because a gate npz without its .pt is an arm you cannot ship, and the best upgrade we
# ever measured was lost exactly that way. A size check is not a check either - truncation can satisfy
# it - so this compares md5 computed on the box against md5 computed here, and refuses quietly to
# report success if they differ.
set -u
PORT="$1"; IP="$2"; DEST="$3"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -p $PORT root@$IP"
mkdir -p "$DEST"
# flatten to ONE line: a multi-line list splits the remote md5sum into separate
# commands, so only the LAST file is hashed while scp copies them all - every file
# then looks collected and only one was actually verified.
FILES=$($SSH 'ls /workspace/knee/raptor_ft_*.pt /workspace/knee/raptor_gold_*.npz /workspace/knee/raptor_ft_*.json 2>/dev/null' 2>/dev/null | tr -d '\r' | tr '
' ' ')
[ -z "$FILES" ] && { echo "  nothing to collect on $IP:$PORT"; exit 0; }
$SSH "md5sum $FILES" 2>/dev/null | grep -v "Warning:\|Welcome\|Have fun" > "$DEST/_remote.md5"
for f in $FILES; do
  scp -o StrictHostKeyChecking=no -P "$PORT" "root@$IP:$f" "$DEST/" >/dev/null 2>&1
done
OK=0; BAD=0
while read -r h p; do
  b=$(basename "$p")
  l=$(md5sum "$DEST/$b" 2>/dev/null | cut -d' ' -f1)
  if [ "$h" = "$l" ]; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "  MISMATCH $b remote=$h local=$l"; fi
done < "$DEST/_remote.md5"
echo "  collected $OK file(s) hash-verified, $BAD mismatched -> $DEST"
[ "$BAD" -gt 0 ] && exit 1
exit 0
