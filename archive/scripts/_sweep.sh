#!/bin/bash
# K_EVAL sweep. Two arms, two different questions.
#   coat224_ep24  - a GATE arm: fold 0 was held out of its training, so the 631 studies are UNSEEN.
#                   Both the absolute number and the shape are honest.
#   confdist_swa  - a SHIPPED arm (one half of the 0.941 champion): it trained on everything but the
#                   58 gold studies, so those 631 studies ARE in its training set. The ABSOLUTE number
#                   is contaminated and meaningless; only the SHAPE across K matters, and contamination
#                   applies equally at every K.
# The question is only ever "how much does this arm lose when given fewer windows", never "is this arm
# good" - so contamination does not invalidate the shape.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWEEP_LABELS=/workspace/knee/labels_cd_ext.parquet
export SWEEP_GROUP=/workspace/knee/study_fingerprint.csv
echo "===== GATE ARM coat224_ep24 (unseen 631) ====="
python -u keval_sweep.py /workspace/arms/raptor_ft_coat224_ep24.pt 48,24,12,9,6
echo "===== SHIPPED ARM confdist_swa @384 (shape only) ====="
python -u keval_sweep.py /workspace/arms/raptor_ft_confdist_swa.pt 48,24,12,9,6
echo "===== SWEEP DONE $(date -u +%H:%M:%S) ====="
