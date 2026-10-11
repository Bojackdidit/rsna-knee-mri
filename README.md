<p align="center">
  <img src="https://raw.githubusercontent.com/Bojackdidit/dread-cinematic-banner/master/dread-orbit-teaser.gif" width="100%" alt="banner">
</p>

# RSNA Knee MRI

Our solution for the [RSNA Knee MRI abnormality detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition: twelve findings per study (ACL, MCL, meniscus, osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture), scored by macro ROC-AUC.

Best public score **0.951**. A CoAtNet at 384px reads short stacks of adjacent slices and pools them with a per-finding attention layer. It is rank-averaged with a second 384px checkpoint, and a three-fold 2.5D ConvNeXt reader is mixed in per finding. Studies are cropped to a 140 mm box, served at 320px, 94 windows per study, with anatomical mirroring.

The score moved from 0.945 to 0.951 by changing how the same weights are served, not by training anything new. Putting one of our existing arms through a better serving pipeline was worth **+0.005**. A fresh nine-hour arm was worth **+0.001**. Every submission since has tested a serving change at fixed weights.

## Layout

| Directory | What's in it |
|---|---|
| `training/` | The finetuning loop (`finetune_box.py`) and the GPU throttle it runs under. |
| `data/` | The direct DICOM reader (`fastdcm.py`), the builders that turn studies into 2.5D slice stacks, and scanner fingerprinting. |
| `labels/` | Twelve-finding soft labels distilled from the free-text radiology reports. |
| `evaluation/` | The scanner-grouped out-of-fold gate, arm correlation, and blend weighting. |
| `kernels/` | The Kaggle inference notebooks — the 0.951 champion and the variants tested against it. |
| `tools/` | The builder that assembles an inference notebook from a public source notebook with our arm swapped in. |
| `docs/` | The catalog of what has been tried and closed. |
| `archive/` | Every other experiment script and kernel variant, kept for reference. |

## Data

Weights, the corpus, and the labels are too large for git and live on Kaggle:

- Corpus (2.5D slice stacks): [`knee-raptor-corpus`](https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus) and [`knee-raptor-corpus-ext`](https://www.kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext)
- Soft labels: [`rsna-knee-labels`](https://www.kaggle.com/datasets/dreaddevelopment/rsna-knee-labels)
- Arm weights: attached to each kernel; the champion arms are in [`raptor-knee-widedense`](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

## What each part is worth

Measured on the leaderboard, one change at a time against a score already on the board.

| Change | Read |
|---|---|
| Our arm through the better serving pipeline | 0.941 → 0.946 |
| Adding the second 384px checkpoint | 0.946 → 0.950 |
| Adding the three-fold reader | 0.950 → 0.951 |
| Our arm alone vs the second checkpoint alone, same route and reader | 0.947 vs 0.950 |
| Dropping our arm from the blend | 0.951 → 0.950 |

The fourth row is the uncomfortable one: head to head our arm is three thousandths behind. The fifth row is why it stays in — it still adds a thousandth, because it gets different studies wrong.

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
kernels/champion-coatnet-published-reader.ipynb
```

Backbones are `timm`; inference reads DICOM directly and runs under the competition's limit on two T4s.

`kernels/champion-coatnet-published-reader.ipynb` is the notebook that scored 0.951. The code is
committed as it ran, so it matches the submission; only the one prose cell was rewritten. `tools/port_pub_pipeline.py` is what built it:
it takes a public source notebook, swaps our arm in, drops the upstream prose, and refuses to write
the result if the geometry it would claim does not match the geometry it serves.

`evaluation/board_position.py` reports where a score stands. It prints the number of teams strictly
above, because that needs no assumption about how a shared score is ordered, and reads our own rank
from the published board rather than inferring it.
