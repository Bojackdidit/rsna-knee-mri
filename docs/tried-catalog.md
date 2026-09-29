# RSNA Knee — Definitive "Already Tried" Catalog (2026-09-29)
Built from a full read of all ~55 knee memory files + the Sept ANS log. READ THIS before proposing or
running anything, so we never backtrack. Champion **0.945**; leaders 0.957/0.959. Board noise ±0.002-0.003.

## DO NOT RE-RUN — closed with evidence
- **Backbones:** CoAtNet-384 (champion, saturated 0.921-0.926), ConvNeXt-B-384 (partner, KEEP),
  ConvNeXt-S (board 0.906), EffV2M (pair 0.942), EffV2S (solo 0.908), ViT-B (0.914), Swin (rho 0.954,
  2x cost), MaxViT (converged weak 0.874-0.888), Raptor SSL base (board 0.833), DINOv3-base (pair 0.942),
  RadImageNet (0.814), OrthoDiffusion (-0.008).
- **Ensembling (EXHAUSTED, all <=0.945):** 2-arm arch-diverse = 0.945 (the one lever, saturates at 2);
  3-arm trio 0.943; CoAtNet folds 0.929; diverse folds 0.932; per-finding weights/routing (board 0.943);
  weight soup (collapses); SWA (dead); prob-mean < rank-mean.
- **Labels (CLOSED + Dread-forbidden):** E[s] collapse, Opus hand-labels, public blend+silence repair,
  Qwen-32B re-extract, our pseudo-labels — all refuted; label-gold-AUC anti-predictive 4/4.
- **Augmentation:** standard stack (+0.015, already in champion); aug-finetune of champions (-0.008, board
  0.936); aug_scanner (NEUTRAL +0.0007 on the correct scanner-grouped OOF, 2026-09-29).
- **Resolution:** 224 caps 0.928 on our recipe; raising res is wrong-direction.
- **Windows/geometry:** K 6/9/12/24/48/72 all measured (48 champion); centre-sampling hurts;
  straddle costs 0; interleaving 0.940; series 3~5, 2-series 0.890; slot_emb/slice_dilate/centre_frac closed.
- **Backbone LR:** 1e-4 vs 3e-5 @96 = -0.088 (untrained-backbone thesis dead).
- **Serve-side edits:** pooling -0.019, re-spacing -0.020. Must be TRAINED in.
- **Speed axis:** 4.2x against every time; runtime floor is reader+pipeline not backbone.
- **Free negatives:** laterality, cross-finding stacking, ROI-crop-0.70, TTA (+0.001).

## GENUINELY UNTRIED (no evidence of any attempt) — where new effort goes
1. **CROP_MM / FOV sweep** — we sit at ~130-140mm, competitors put the sweet spot at 140-160 (Tom Aindow
   150, Parag 140-160). Never swept. NOTE: must be TRAINED at the new FOV (serve-side crop change = refuted).
2. **Multi-slice-as-channels stem** (in_chans=16, feed a slice stack as channels vs adjacent-RGB triples).
3. **DICOM metadata as model INPUT features** (n_meta) — only ever tested as a leakage shortcut (it isn't
   one: 0.598 scanner-grouped), never wired as features.
4. **A single arm trained to FULL convergence** — every arm capped 14-16 epochs; FULL96 still climbing ep13.
5. **`--pool max` trained-in** (vs the refuted serve-side pooling).
6. 5-fold ON TOP of full-data-strong singles (NOT the weak 80% fold arms that already lost twice).

## HARD CONSTRAINTS (Dread)
NO Vast. NO label work. NO window-count re-litigation. Local GPU only, bounded, ends-in-a-decision, no
speculative burns. Board is the ONLY valid instrument (no offline gate; OOF sd 0.0028 guides only).
5 submits/day, best-of protects, never tread water. Runtime <30min & >0.940. Identity "Vera", plain
descriptions, no third-party datasets/dead-code.

## FRONTIER
Gap = +0.012-0.014, and it is **single-model generalization**, NOT ensembling or labels. Our CoAtNet-384
attention-MIL over a 96-slice/5-series volume OVERFITS (Scott: 0.930 gold->0.949 board; us 0.933 gold->
0.941 board — leaders score LOWER on gold, HIGHER on the hidden test). The public frontier (0.941-0.943)
is Dread's own Raptor arm forked, caps below us; leaders are private. The untried input-representation
levers (FOV #1, channel-stem #2, metadata #3) and a convergence-trained single arm (#4) are the honest
remaining attacks on that overfit.
