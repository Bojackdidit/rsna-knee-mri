#!/bin/bash
# Everything that must happen after fold 4 lands, as ONE detached OS process - the harness
# background tasks kept getting killed (2026-09-26 ~13:00, twice within minutes), while plain OS
# processes survive. Runs: F4 dataset upload -> OOF dump + assembly -> PUBLIC44.
# Output appends to _folds.log so the surviving monitor sees it.
set -u
cd "$(dirname "$0")"

echo "=== post-F4 chain armed $(date +%H:%M) pid $$ ==="

# 1. wait for fold 4's artifacts (json+pt written together at training end)
while [ ! -f "raptor_ft_LR96_F4.json" ] || [ ! -f "raptor_ft_LR96_F4.pt" ]; do sleep 120; done
sleep 30

# 2. upload F4 to the fold-arms dataset, verify by file listing
cp raptor_ft_LR96_F4.pt fold_arms_ds/
for i in 1 2 3; do
  kaggle datasets version -p fold_arms_ds --dir-mode zip -m "add LR96_F4" 2>&1 | tail -1
  sleep 60
  if kaggle datasets files dreaddevelopment/knee-raptor-fold-arms --csv 2>/dev/null | grep -q "LR96_F4"; then
    echo "F4 verified in dataset $(date +%H:%M)"; break
  fi
done

# 3. OOF dump (its own guards wait for the trainer to exit) then the assembly numbers
bash _oof_dump.sh

# 4. the 44-slice public release arm, last
bash _release44_chain.sh

echo "=== post-F4 chain done $(date +%H:%M) ==="
