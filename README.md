<p align="center">
  <img src="https://raw.githubusercontent.com/Bojackdidit/dread-cinematic-banner/master/dread-orbit-teaser.gif" width="100%" alt="banner">
</p>

# RSNA Knee MRI

Our solution for the [RSNA Knee MRI abnormality detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition: twelve findings per study (ACL, MCL, meniscus, osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture), scored by macro ROC-AUC.

Best public score **0.945**. Two backbones — a CoAtNet and a ConvNeXt at 384px — read short stacks of adjacent slices, pool them with a per-finding attention layer, and combine by rank average. The two are trained on the same data but disagree enough that the pair beats either alone.

## Layout

| Directory | What's in it |
|---|---|
| `training/` | The finetuning loop (`finetune_box.py`) and the GPU throttle it runs under. |
| `data/` | The direct DICOM reader (`fastdcm.py`), the builders that turn studies into 2.5D slice stacks, and scanner fingerprinting. |
| `labels/` | Twelve-finding soft labels distilled from the free-text radiology reports. |
| `evaluation/` | The scanner-grouped out-of-fold gate, arm correlation, and blend weighting. |
| `kernels/` | The Kaggle inference notebooks — the 0.945 champion and the variants tested against it. |
| `docs/` | The catalog of what has been tried and closed. |
| `archive/` | Every other experiment script and kernel variant, kept for reference. |

## Data

Weights, the corpus, and the labels are too large for git and live on Kaggle:

- Corpus (2.5D slice stacks): [`knee-raptor-corpus`](https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus) and [`knee-raptor-corpus-ext`](https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext)
- Soft labels: [`rsna-knee-labels`](https://www.kaggle.com/datasets/dreaddevelopment/rsna-knee-labels)
- Arm weights: attached to each kernel; the champion arms are in [`raptor-knee-widedense`](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

## Reproduce

```
# 1. build the corpus from the competition DICOM
python data/knee_corpus_v11.py

# 2. train an arm (grouped folds hold out whole scanners)
python training/finetune_box.py --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --pool attn --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10

# 3. score it out-of-fold on the scanner-grouped split
python evaluation/gate_eval.py raptor_ft_<tag>.pt <tag>

# 4. ship: the inference notebook loads the arm weights and predicts
kernels/champion-coatnet-plus-convnext.ipynb
```

Backbones are `timm`; inference reads DICOM directly and runs under the competition's one-hour limit on two T4s.
