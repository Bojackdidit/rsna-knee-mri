#!/bin/bash
# Wait for the corpus download+unzip to finish, verify it, then start the screen. No idle gap.
cd /workspace/knee
for i in $(seq 1 240); do
  if [ ! -f corpus/knee-raptor-corpus.zip ] && [ ! -f corpus/knee-raptor-corpus.zip.kaggle-partial ]; then
    n=$(find corpus -maxdepth 2 -type d 2>/dev/null | wc -l)
    [ "$n" -gt 2 ] && { echo "CORPUS READY at $(date -u +%H:%M:%S), $n dirs"; break; }
  fi
  sleep 30
done
du -sh corpus 2>/dev/null; ls corpus | head -5; find corpus -name "*.npy" -o -name "*.npz" -o -name "*.pt" 2>/dev/null | wc -l
bash _run_effnet.sh
