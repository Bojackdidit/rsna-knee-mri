#!/bin/bash
# Fire the 224 pair the moment the Kaggle day rolls over. Dread asked for the 224 work on the
# board; the slot for today was already spent, so this waits for the reset rather than losing
# another day. Pair first, not the trio - read one result before deciding the next.
set -u
LOG=/d/claude2/main_track/knee/submit_224.log
TARGET=$(date -u -d "tomorrow 00:00:30" +%s 2>/dev/null || date -u -v+1d -v0H -v0M -v30S +%s)
echo "$(date -u) armed; firing at $(date -u -d @$TARGET 2>/dev/null || echo $TARGET)" >> "$LOG"
while [ "$(date -u +%s)" -lt "$TARGET" ]; do sleep 60; done
cd /d/claude2/main_track/knee
kaggle competitions submit -c rsna-knee-abnormality-detection \
  -k dreaddevelopment/knee-mri-raptor-224-pair -v 1 -f submission.csv \
  -m "Two CoAtNets averaged by rank, both at 224 pixels instead of the 384 we normally run. Several top placements report a single model at 224 scoring around 0.934 at a fraction of the compute, so this tests that setting directly on our pipeline. These are the best two of five models we trained at 224, one on labels distilled from an earlier model and one on those labels averaged with the public set. Everything else matches the entry that scores 0.941, so resolution is the only change. Each model reads about a third of the pixels, so this should finish well inside the 29 minutes the 384 pair takes." \
  >> "$LOG" 2>&1
echo "$(date -u) submit returned $?" >> "$LOG"
tail -3 "$LOG"
