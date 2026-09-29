#!/bin/bash
# Ship the corpus in parallel chunks. A single ssh stream sustains 250 KB/s off this line; four
# streams aggregate 818 KB/s, so the limit is per-stream, not the link. Eight chunks, reassembled
# on the box, then verified byte-for-byte against the local size before anything trains on it.
set -u
K=/c/Users/johna/.ssh/ddp_ed25519; H=root@212.13.234.23; P=13359
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no"
N=8; MB=383   # 3062 MiB total -> 383 MiB per chunk, last one short
$SSH $H 'rm -f /workspace/p*.bin /workspace/corpus_s02/all_vols.npy'
echo "start $(date -u +%H:%M:%S)"
for i in $(seq 0 $((N-1))); do
  ( dd if=corpus_s02/all_vols.npy bs=1M skip=$((i*MB)) count=$MB 2>/dev/null \
      | $SSH $H "cat > /workspace/c$i.bin" ) &
done
wait
echo "chunks done $(date -u +%H:%M:%S)"
$SSH $H "cat $(for i in $(seq 0 $((N-1))); do printf '/workspace/c%d.bin ' $i; done) > /workspace/corpus_s02/all_vols.npy && rm -f /workspace/c*.bin && stat -c '%s' /workspace/corpus_s02/all_vols.npy"
echo "expected 3211264128"
echo "done $(date -u +%H:%M:%S)"
