# RSNA Knee MRI, our solution

Twelve findings per study, scored by macro ROC-AUC.

Final: **977th of 6,425, private 0.942**, up 290 places from a public rank of 1267. The private
split is about 70% of the test data.

Two entries were selected. One scored only our own weights. The other put our arm into the
configuration most of the field was running, and it is the one that counted.

## The two entries

| Entry | Public | Private | What scores in it |
|---|---|---|---|
| `kernels/final-raptor-pair.ipynb` | 0.947 | 0.930 | Two of our arms, rank-averaged at equal weight. No published checkpoint, no reader, no fitted weights. |
| `kernels/final-spine-with-raptor-arm.ipynb` | 0.957 | **0.942** | The published CoAtNet at 384 and at a 0.8 crop to 224, a public four-fold ConvNeXt, and a third arm of ours, `CONFDIST_K24_SWA`, entering the finished per-finding rank at 0.08. |

They share no scoring checkpoint. The arm in the second entry is not either of the two in the
first, which is why.

The scored run of the first entry attached two datasets, `raptor-knee-scratch-aug2` and
`raptor-knee-convnext-arm`. Both are ours. The notebook still carries a dormant reader stage
inherited from the public notebook it was built on; with no reader weights attached it prints
`Reader unavailable` and passes the pair through unchanged.

The second entry carries fixed weights the first does not: per-finding weights across the Raptor
half, and 0.35 of that half to the 224 model. All were set before the run, none are fitted at test
time.

## Corpus

Each study is five MRI series. We build one 96-slice volume per study with a fixed slot layout, so
every study presents the same shape regardless of which sequences it actually has.

| Slot | Plane | Fluid sensitive | Slices |
|---|---|---|---|
| 0 | Sagittal | yes | 27 |
| 1 | Sagittal | no | 21 |
| 2 | Coronal | yes | 18 |
| 3 | Coronal | no | 12 |
| 4 | Axial | either | 18 |

Slices are taken evenly across the middle 96% of each series. Each is cropped to a 140 mm physical
box using the DICOM pixel spacing, then resized to 336 px with INTER_AREA. Intensity is normalised
per slot on its own 2nd and 98th percentiles and stored as uint8. A missing series leaves its slot
zeroed and masked rather than borrowing slices from another plane.

4,407 studies. Physical cropping matters more than it sounds. Pixel spacing varies by scanner, so
a fixed pixel crop gives the model a different anatomical field of view per study. Cropping in
millimetres makes the field of view constant.

The slice count and the span move together, and only together. Widening the span from the middle of
each series out to 2-98% while holding 44 slices scored 0.917, down from the 0.926 it started at.
The same wider span with 64 slices scored 0.928. Spreading the same number of slices thinner loses
more than the extra coverage gains, so density has to rise with span. The 96-slice layout above is
that lever taken further.

Builder is `data/knee_corpus_v11.py`. An earlier 44-slice version of this corpus is published as
`knee-raptor-corpus` and `knee-raptor-corpus-ext`.

## Labels

The competition ships 58 expert-labelled studies. Everything else we labelled ourselves, parsing
each study's radiology report with an LLM into a calibrated probability per finding rather than a
hard 0 or 1. Soft targets train better here because they carry the report's own uncertainty, and a
radiologist hedging in prose is information the threshold throws away.

These are published as `rsna-knee-labels` under CC0.

The scoring arms trained on a further derivative, `labels_confdist.parquet`. Its builder is not in
this repository, which only reads the file. The distillation code in `labels/` builds candidate
successors to it: an improved label blend supplies the ordering within each finding, and that
ordering is mapped onto the distribution of the previous label set, keeping the better ranking
without moving the overall prevalence that the loss weights depend on.

The 58 expert studies were never trained on. They were used for epoch selection only.

## Architecture

A timm backbone reads three-slice windows as three-channel images. A per-finding attention head
pools across all windows of a study.

Each finding gets its own attention vector over the window bag and its own class vector, so a
finding can concentrate on the windows where it is visible. Patellofemoral osteoarthritis and ACL
tears do not live in the same slices, and a single pooled study vector forces them to share one.

Head is `RaptorClassifier` in `training/finetune_box.py`.

Loss is BCE with logits. The positive weight per finding comes from its prevalence in the soft
labels, clipped so no finding dominates.

## Training

Both scoring arms use identical commands apart from the backbone.

```
python finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \   # or convnext_base.fb_in22k_ft_in1k_384
  --res 384 --ckpt timm --norm imagenet \
  --labels labels_confdist.parquet \
  --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 4242 \
  --aug geo,erase,blur,noise --ema 0.9998
```

24 windows per study per step, batch of 2 studies, gradient checkpointing to fit 384 px activations.
Separate learning rates for backbone and head. EMA at 0.9998. Augmentation is geometric
jitter, random erasing, blur and noise. No flips during training, because a left and right knee are
not interchangeable and a naive flip teaches the model that they are.

| Arm | Backbone | gold-58 |
|---|---|---|
| SCRATCH_AUG2 | CoAtNet-RMLP-2 384 | 0.9303 |
| CONVNEXT_E24 | ConvNeXt-Base 384 | 0.9270 |

The pair is architecturally diverse and within 0.0033 of each other. A partner more than about
0.0055 weaker contributed nothing in every pairing we measured.

## Serving

The pure entry is served through the public engine's geometry, not the geometry it trained at.
That stage builds native-384 volumes with slot counts of 26, 22, 18, 12 and 18 and centre-crops
each window to 320 px. The corpus table above describes training.

94 three-slice windows per study, every position straight down the stack. Eight of the 94 straddle
a join between two series and mix them in their three channels. Training keeps windows inside one
slot; restricting serving the same way was measured and read 0.957 against the 0.958 it changed, so
the straddling windows ship.

Anatomical mirroring, not a width flip. A sagittal window mirrors by reversing its three slice
channels, since the slice axis is the medial to lateral axis. Coronal and axial windows mirror
across image width. The transform is applied before normalisation. A plain width flip on a sagittal
window is not a mirror of the knee and measured worse.

The two arms are combined as per-finding percentile ranks at equal weight. Ranking per finding
rather than averaging probabilities means the two models only have to agree on ordering, not on
calibration.

## What our own weights cost

Measured one step at a time against the same public subset.

| Share of the final rank held by our weights | Public | Private |
|---|---|---|
| 100% | 0.947 | 0.930 |
| 25% | 0.956 | 0.941 |
| 8% | 0.957 | 0.942 |
| 0% | 0.958 | 0.942 |

Monotonic on both. The middle three rows share one pipeline; the 100% row cannot, since an entry
made only of our weights has no spine to sit in.

We selected the pure entry anyway, because it carries no fitted weights and nothing in it could be
tuned to the public split. The private board did not reward that, and the next section says so.

## What the private board showed

Three results, none of which we predicted correctly.

**Our arm at 8% was exactly neutral.** The spine entry and the untouched public configuration both
finished on 0.942. Not better, not worse, identical at three decimals. The public gap of 0.957
against 0.958 had already said that, and it transferred exactly.

**The unfitted entry regressed more, not less.** This is the one worth sitting with. The argument
for the pure entry was that it carries no fitted parameters, so nothing in it could have been tuned
to the public split and it should give back less.

| Entry | Public | Private | Drop |
|---|---|---|---|
| pure pair, zero fitted weights | 0.947 | 0.930 | 0.017 |
| public configuration, hand-tuned per-finding weights | 0.958 | 0.942 | 0.016 |
| spine with our arm at 0.08 | 0.957 | 0.942 | 0.015 |

The entry with nothing to overfit fell furthest. Being unfitted did not protect it. Whatever drove
the near-uniform 0.015 to 0.017 drop across the whole board, it was not the hand-tuned weights,
and the reasoning that selected the pure entry was wrong on its own terms.

**The whole board compressed rather than reordered.** Public top 0.969, private top 0.960. Our 290
places came from churn inside a large tie at 0.942 and from teams above falling further, not from
our model separating itself. The top did shake: the public leader finished 6th.

## What did not help

### External data from OAI

The Osteoarthritis Initiative holds 4,791 subjects with knee MRI. We converted 5,178 knees into our
exact corpus geometry and trained with masked supervision, where external studies carry loss only on
the cells they can label.

OAI grades come from a fixed-flexion PA radiograph, which can grade joint space narrowing in the
medial and lateral compartments and nothing else. So only two of twelve findings receive
supervision. Patellofemoral osteoarthritis is not labelled, because that radiograph cannot see the
patellofemoral compartment at all.

Lateral OA was our weakest finding at 0.807, and it is one of the two cells OAI can supervise. That
was the whole case for doing it.

Both supervised cells came out worse.

| Finding | With OAI | Without | Change |
|---|---|---|---|
| Lateral OA (supervised) | 0.774 | 0.807 | −0.033 |
| Medial OA (supervised) | 0.980 | 0.991 | −0.011 |
| macro | 0.9161 | 0.9277 | −0.0116 |

Lateral OA read exactly 0.774 at four consecutive epoch checkpoints, so this is not a noisy draw on
one reading. Eleven of twelve findings finished worse or flat. Patellofemoral osteoarthritis, at
+0.006, was the only one that improved, and it is the one finding OAI never labelled.

The pipeline was sound. 5,178 knees converted cleanly at a mean of 96.0 of 96 slices filled, every
slot carrying real signal, 56% of label cells trainable. The lever was wrong, not the plumbing.

The reason it fails is worth stating plainly. External studies consume optimizer steps while
contributing gradient to two cells out of twelve. The other ten findings get diluted, and the two
that receive supervision are taught from a radiograph that disagrees with what an MRI shows. A
larger training set is not automatically a better one.

Weights and the masked label tables are preserved as `raptor-arm-oai-ext` and
`knee-oai-masked-supervision-tables`.

### Other measured failures

| Change | Read |
|---|---|
| Serving our arms at 336 px rather than the 384 route | 0.947 → 0.945 |
| Substituting our arm for the published 384 checkpoint outright | 0.958 → 0.956 |
| Raising our arm's share of the finished rank from 0.08 to 0.25 | 0.957 → 0.956, on the four-model variant. The no-reader twin at 0.25 errored on rerun and holds no score. |
| Width-flip TTA on sagittal windows instead of channel reversal | measured worse, not shipped |

The first one is the uncomfortable result. We expected the geometry the arms were trained at to beat
the published route, and it did not.

## Limitations

The 58 expert studies are the only labels scored the same way as the leaderboard. They were used for
epoch selection, so they are a development instrument and not an independent estimate. Nine separate
predictions from that set failed to transfer to the board.

Public score differences of a thousandth do not establish anything. The public subset is roughly 400
studies. We report them because they are what we measured, not because each one is a real gain.

Every number above is a public leaderboard read unless stated otherwise.

## Credits

The competition data comes from RSNA and its contributors.

The 0.957 entry runs nartaa's published SWA checkpoints alongside a public four-fold ConvNeXt.

OAI data was obtained from the NIMH Data Archive. Dataset identifier 10.15154/0hcg-f676.

## Repository

| Directory | Contents |
|---|---|
| `training/` | The finetuning loop and the GPU throttle it runs under. |
| `data/` | The direct DICOM reader and the corpus builders. |
| `labels/` | Report-derived soft labels and the distillation step. |
| `evaluation/` | Out-of-fold scoring, arm correlation, blend weighting. |
| `kernels/` | The two selected entries and the variants tested against them. |
| `docs/` | What has been tried and closed. |

Weights and the corpus are too large for git and live on Kaggle. Links are in the README.
