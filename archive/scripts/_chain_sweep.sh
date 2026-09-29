#!/bin/bash
# Wait for the LR run on this box to exit, then spend the otherwise-idle GPU on Phase 0a.
#
# The sweep asks the one question the grouped gate is actually VALID for: same arm, same labels, only
# the serve geometry changes. Windows are currently linspaced across the whole slice range, which is
# fine at K=48 and wasteful at K=9-12 - most of the few windows we can afford land on the ends of the
# stack. Concentrating them costs nothing at inference and needs no retraining.
#
# Uses the CHAMPION arm (confdist_swa, md5 95855c08fa7c12abbf52dce039e275c4, pulled from our own Kaggle
# dataset and hash-matched against the local copy), so the result transfers directly to the kernel that
# actually scores 0.941.
#
# Process detection never uses pkill/pgrep -f: this script's own command line contains the pattern and
# would self-match - that trap once reported two running trainers when none existed.
set -u
cd /workspace/knee
while true; do
  N=$(ps -eo args | grep -ac "[f]inetune_box.py")
  [ "$N" -eq 0 ] && break
  sleep 60
done
echo "trainer exited $(date -u +%H:%M:%S) - starting sweep" >> /workspace/sweep_chain.log
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export SWEEP_LABELS=/workspace/knee/labels_cd_ext.parquet
export SWEEP_GROUP=/workspace/knee/study_fingerprint.csv
python -u keval_centre_sweep.py /workspace/knee/champ/raptor_ft_confdist_swa.pt "12,9" "1.0,0.75,0.5,0.35" \
  > /workspace/knee/sweep_out.log 2>&1
echo "SWEEP DONE $(date -u +%H:%M:%S)" >> /workspace/sweep_chain.log
