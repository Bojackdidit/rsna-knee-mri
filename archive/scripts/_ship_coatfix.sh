#!/bin/bash
# One command to ship the retrained CoAtNet once its weights are local. Usage: _ship_coatfix.sh TAG [TAG2]
# Steps: stage weights -> create/version the dataset -> build the kernel -> preflight -> print next action.
# Nothing here submits; the last step is deliberately manual.
set -u
cd /d/claude2/main_track/knee || exit 1
A1="$1"; A2="${2:-}"
for T in $A1 $A2; do
  [ -f "arms_spec/raptor_ft_${T}.pt" ] || { echo "missing arms_spec/raptor_ft_${T}.pt - run _pull_arm.sh $T first"; exit 1; }
  cp "arms_spec/raptor_ft_${T}.pt" dsup_coatfix/
done
echo "staged: $(ls -1 dsup_coatfix/*.pt | wc -l) weight file(s)"
if kaggle datasets files dreaddevelopment/raptor-knee-coat384-fix >/dev/null 2>&1; then
  kaggle datasets version -p dsup_coatfix -m "retrained CoAtNet-384 arms, three series" -d 2>&1 | tail -2
else
  kaggle datasets create -p dsup_coatfix 2>&1 | tail -2
fi
ARGS="arms_spec/raptor_ft_${A1}.pt"
[ -n "$A2" ] && ARGS="$ARGS arms_spec/raptor_ft_${A2}.pt"
rm -rf variants/push_knee_mri_coatnet_fixed
FASTARM_SRC=variants/push_fastdcm_clean python build_fastarm.py \
  knee-mri-coatnet-fixed dreaddevelopment/raptor-knee-coat384-fix \
  coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k 384 12 $ARGS \
  --slots 0,2,4 --slotwin --title "Knee MRI - CoAtNet Fixed" 2>&1 | tail -16
echo
python preflight_push.py variants/push_knee_mri_coatnet_fixed --live 2>&1 | tail -12
echo
echo "if preflight says OK:  kaggle kernels push -p variants/push_knee_mri_coatnet_fixed"
