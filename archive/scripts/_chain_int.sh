#!/bin/bash
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export SWEEP_LABELS=/workspace/knee/labels_cd_ext.parquet
export SWEEP_GROUP=/workspace/knee/study_fingerprint.csv
until grep -q "best per K" cwsweep.log 2>/dev/null; do sleep 30; done
echo "===== PLACEMENT SWEEP DONE, starting interleave ====="
python -u interleave_test.py /workspace/arms/raptor_ft_confdist_swa.pt,/workspace/arms/raptor_ft_coatnet_w80_full_swa.pt 12,9,6
echo "===== INTERLEAVE DONE $(date -u +%H:%M:%S) ====="
