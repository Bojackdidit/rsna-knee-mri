#!/bin/bash
cd /workspace/knee
until grep -q "INTERLEAVE DONE" interleave.log 2>/dev/null; do sleep 45; done
bash _run_dilate.sh
