#!/bin/bash
# FINE-TUNE the two shipped champion arms with augmentation. Not a retrain.
#
# The champion boards 0.941 and was trained with NO augmentation - finetune_box.py's default was a
# +/-10% gain and nothing else, so every shipped arm is effectively unaugmented. A controlled A/B
# measured a standard stack at +0.0150 arm-level, the largest single lever ever found here, and it
# has never been applied to the arms we actually ship.
#
# So: start from each champion's OWN weights (--init_arm, built for exactly this), turn augmentation
# on, and run a few epochs at a LOW learning rate. This adapts the arm; it does not overwrite it.
# It also lands the arm at the 96-slice geometry the kernel actually serves, which the shipped
# weights have never seen at training time.
#
# bs 2, not 8: measured tonight that bs 4 with grad_ckpt at k=24/384 peaks near 19.8 GiB and OOMs a
# 24 GiB card. bs counts STUDIES; the BatchNorms still see bs*k = 48 images per step.
set -u
C="${1:?contract id}"
K=/c/Users/johna/.ssh/id_ed25519
LOG=_bringup_ft.log
exec > >(tee -a "$LOG") 2>&1
echo "=== finetune bringup $(date -u +%H:%M:%S) contract $C ==="

H=""; P=""
for i in $(seq 1 120); do
  J=$(vastai show instance "$C" --raw 2>/dev/null)
  H=$(echo "$J" | python -c "import sys,json;print(json.load(sys.stdin).get('public_ipaddr') or '')" 2>/dev/null)
  P=$(echo "$J" | python -c "import sys,json;d=json.load(sys.stdin);print((d.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort') or '')" 2>/dev/null)
  S=$(echo "$J" | python -c "import sys,json;print(json.load(sys.stdin).get('actual_status') or '')" 2>/dev/null)
  if [ -n "$P" ] && [ "$S" = "running" ]; then echo "[ssh] $H:$P status=$S"; break; fi
  echo "[wait $i] status=$S port=${P:-none}"; sleep 10
done
[ -n "$P" ] || { echo "ABORT: no ssh port after 20 min"; exit 1; }
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 root@$H"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
for i in $(seq 1 60); do $SSH 'echo up' 2>/dev/null | grep -q up && break; echo "[sshwait $i]"; sleep 10; done
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -BG --output=avail /workspace 2>/dev/null | tail -1'

echo "[stage] code, labels, credentials"
$SSH 'mkdir -p /workspace/knee/arms /root/.kaggle'
$SCP finetune_box.py _merge_v11.py train.csv study_fingerprint.csv root@$H:/workspace/knee/
$SCP phase2/box_stage/labeler/labels_confdist.parquet root@$H:/workspace/knee/
$SCP /c/Users/johna/.kaggle/kaggle.json /c/Users/johna/.kaggle/access_token root@$H:/root/.kaggle/ 2>/dev/null
$SSH 'chmod 600 /root/.kaggle/* 2>/dev/null; pip -q install kaggle timm pandas pyarrow scikit-learn 2>&1 | tail -1; kaggle --version'

echo "[stage] champion checkpoints (293 MB each) - verify by hash, a truncated arm is silent"
$SCP phase2/ship_confdist/raptor_ft_confdist_swa.pt root@$H:/workspace/knee/arms/
$SCP soup/raptor_ft_coatnet_w80_full_swa.pt root@$H:/workspace/knee/arms/
for f in raptor_ft_confdist_swa.pt raptor_ft_coatnet_w80_full_swa.pt; do
  L=$(md5sum $( [ "$f" = "raptor_ft_confdist_swa.pt" ] && echo phase2/ship_confdist/$f || echo soup/$f ) | cut -d' ' -f1)
  R=$($SSH "md5sum /workspace/knee/arms/$f | cut -d' ' -f1")
  [ "$L" = "$R" ] && echo "  $f md5 MATCH $L" || { echo "  $f MD5 MISMATCH local=$L box=$R"; exit 1; }
done

echo "[corpus] assembling the 96-slice corpus (the geometry the kernel actually serves)"
$SSH 'cd /workspace/knee && setsid nohup python -u _merge_v11.py --out /workspace/knee/corpus_v11 > merge.log 2>&1 < /dev/null & echo started'
for i in $(seq 1 200); do
  D=$($SSH 'cd /workspace/knee; ls corpus_v11/all_ids.npy >/dev/null 2>&1 && echo DONE || grep -c placed merge.log 2>/dev/null' 2>/dev/null)
  echo "[corpus $i] $D"
  [ "$D" = "DONE" ] && break
  sleep 60
done
$SSH 'cd /workspace/knee && python -c "
import numpy as np
v=np.load(\"corpus_v11/all_vols.npy\",mmap_mode=\"r\")
assert v.shape==(4407,96,336,336), v.shape
s=v[::100]; nz=(s.reshape(len(s),-1).sum(1)>0).mean()
assert nz>0.99, (\"mostly zeros\", nz)
print(\"CORPUS VERIFIED\", v.shape, \"nonzero %.4f\"%nz)
"' || { echo "ABORT: corpus verification failed"; exit 1; }

FT="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
--group_csv /workspace/knee/study_fingerprint.csv --group_folds 5 --group_fold 0 --reclaim 0.7 \
--epochs 6 --bb_lr 1e-5 --head_lr 1e-4 --workers 4 --aug geo,erase,blur,noise"

$SSH "cat > /workspace/knee/ft_chain.sh <<'EOS'
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
run () {  # run <tag> <source arm>
  echo \"\$1 from \$2 start \$(date -u +%F\ %H:%M:%S)\" >> ft_chain.log
  python -u finetune_box.py $FT --init_arm /workspace/knee/arms/\$2 --tag \$1 > run_\$1.log 2>&1
  echo \"\$1 done rc=\$? \$(date -u +%F\ %H:%M:%S)\" >> ft_chain.log
}
run FT_CONFDIST raptor_ft_confdist_swa.pt
run FT_W80      raptor_ft_coatnet_w80_full_swa.pt
echo \"FT CHAIN COMPLETE \$(date -u +%F\ %H:%M:%S)\" >> ft_chain.log
EOS
chmod +x /workspace/knee/ft_chain.sh
cd /workspace/knee && setsid nohup bash ft_chain.sh > ft_outer.log 2>&1 < /dev/null & echo launched"
sleep 90
$SSH 'cd /workspace/knee; echo "trainers: $(pgrep -f "[f]inetune_box.py" | wc -l)"; head -22 run_FT_CONFDIST.log 2>/dev/null'
echo "=== bringup done $(date -u +%H:%M:%S) | ssh -i $K -p $P root@$H ==="
