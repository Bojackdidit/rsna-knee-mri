#!/bin/bash
# Take one finished architecture arm from its box to a pushed kernel, ready to submit.
#
# The submit itself is deliberately NOT here: slots are five a day and which arm is worth one is a
# judgement made after reading the correlation against the champion, not something a script should
# decide. Everything before that point is mechanical and is done here so three arms landing within
# hours of each other do not each cost an hour of hand-work.
#
# Usage: stage_arm.sh <TAG> <ARCH> <PORT> <IP> <DATASET-SLUG> <DATASET-TITLE> <KERNEL-REF> <KERNEL-TITLE>
set -euo pipefail
TAG="${1:?tag}"; ARCH="${2:?arch}"; PORT="${3:?port}"; IP="${4:?ip}"
DS="${5:?dataset slug}"; DSTITLE="${6:?dataset title}"; KREF="${7:?kernel ref}"; KTITLE="${8:?kernel title}"
SSH="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP"
DIR="ship_${TAG,,}"

echo "== pulling $TAG from $IP:$PORT"
mkdir -p "$DIR" gold_oof
scp -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$PORT" \
  "root@$IP:/workspace/knee/raptor_ft_${TAG}.pt" "$DIR/raptor_ft_${TAG}.pt"
scp -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$PORT" \
  "root@$IP:/workspace/knee/raptor_gold_${TAG}.npz" "gold_oof/raptor_gold_${TAG}.npz"

# A size check is satisfied by a truncation. Hash both ends.
REMOTE=$($SSH "md5sum /workspace/knee/raptor_ft_${TAG}.pt" 2>/dev/null | awk '{print $1}')
LOCAL=$(md5sum "$DIR/raptor_ft_${TAG}.pt" | awk '{print $1}')
[ "$REMOTE" = "$LOCAL" ] || { echo "MD5 MISMATCH remote=$REMOTE local=$LOCAL"; exit 2; }
echo "   md5 $LOCAL verified both ends"

echo "== dataset $DS"
cat > "$DIR/dataset-metadata.json" <<JSON
{
  "title": "$DSTITLE",
  "id": "$DS",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON
if kaggle datasets status "$DS" >/dev/null 2>&1; then
  kaggle datasets version -p "$DIR" -m "Arm $TAG, $ARCH." -r skip
else
  kaggle datasets create -p "$DIR" -r skip
fi

# A kernel that starts while the version is still processing silently mounts the PREVIOUS one, and
# nothing in its log can tell you afterwards. Wait for ready.
for i in $(seq 1 60); do
  S=$(kaggle datasets status "$DS" 2>&1 | tr -d '[:space:]')
  echo "   $(date -u +%H:%M:%S) dataset status: $S"
  [ "$S" = "ready" ] && break
  sleep 20
done
[ "$S" = "ready" ] || { echo "dataset never became ready"; exit 3; }

echo "== kernel $KREF"
python make_pair_variant.py --tag "$TAG" --arch "$ARCH" --dataset "$DS" --kernel "$KREF" --title "$KTITLE"
kaggle kernels push -p "variants/push_pair_${TAG,,}"
echo "== staged. Check the arm's correlation with _arm_corr.py before spending a slot."
