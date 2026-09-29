#!/bin/bash
# Full pipeline, no handicaps: ALL 4407 RSNA studies + ALL 1124 MRNet, holdout 0, full 16 epochs.
set -eu
cd /workspace
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "deps"
pip install -q "numpy<2" kaggle pydicom timm pandas pyarrow opencv-python-headless scikit-learn 2>&1 | tail -1
pip install -q "numpy<2" 2>&1 | tail -1
python -c "import numpy,cv2,sklearn,pydicom,timm,torch;print('numpy',numpy.__version__,'cv2',cv2.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"

log "download competition data (parallel range requests)"
python -u _pardl.py
ls -la /workspace/rsna-knee-abnormality-detection.zip

log "download MRNet external data"
kaggle datasets download techtech69/mrnet-knee-mri-stanford-aimi -p /workspace/ext 2>&1 | tail -1
log "download labels"
kaggle datasets download dreaddevelopment/rsna-knee-labels -p /workspace/labels --unzip 2>&1 | tail -1
python -c "
import zipfile; z=zipfile.ZipFile('/workspace/rsna-knee-abnormality-detection.zip'); z.extract('train.csv','/workspace/')
print('train.csv extracted')"

log "build RSNA v11 corpus - ALL studies"
mkdir -p corpus
python -u build_raptor_corpus.py --zip /workspace/rsna-knee-abnormality-detection.zip --out /workspace/corpus --workers 32 2>&1 | tail -3

log "build MRNet v11 - fixed slots, no blanks"
mkdir -p mrnet
python -u build_raptor_mrnet2.py --zip /workspace/ext/mrnet-knee-mri-stanford-aimi.zip --out /workspace/mrnet 2>&1 | tail -3
python -c "
import numpy as np
m=np.load('/workspace/mrnet/raptor_mrnet_masks.npy')
print('MRNet valid positions per study: mean %.1f of 96 (old broken build was 63)'%m.sum(1).mean())
assert m.sum(1).mean()>90, 'MRNet slots still blank - refusing to train on a handicapped corpus'"

log "combine - holdout 0, nothing withheld"
mkdir -p corpus_all
cp corpus/raptor_v11_* corpus_all/
cp mrnet/raptor_mrnet_* corpus_all/
python -u combine_raptor_corpus.py --corpus /workspace/corpus_all --confdist /workspace/labels/labels_llm_soft.parquet --out /workspace/combined --holdout 0 2>&1 | tail -8

log "patch trainer for crash-safe per-epoch checkpoints"
cp finetune_raptor_cnn8.py finetune_raptor_cnn9.py
python patch_save_each_epoch.py finetune_raptor_cnn9.py finetune_raptor_cnn9.py

log "TRAIN - full data, 16 epochs"
export KNEE_DATA_ROOT=/workspace/combined
setsid nohup python -u finetune_raptor_cnn9.py --ckpt timm --norm imagenet --workers 16 \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --slots "" --k 12 --k_eval 24 \
  --epochs 16 --bs 8 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777 \
  --tag raptor_ext_full --labels /workspace/combined/labels_train.parquet \
  > /workspace/train_ext_full.log 2>&1 < /dev/null &
log "training launched"
