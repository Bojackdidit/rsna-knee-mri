#!/usr/bin/env python3
"""SOLO arm, both GPUs. A speed-only change: the predictions must not move.

The solo kernel loads one model on cuda:0 and leaves the second card idle, because the
concurrency in the shipped kernel is one ARM per GPU and a solo run only has one arm.
DICOM reading is ~73% of the clock, so this cannot halve the run - but the GPU half of it
is currently running on one card instead of two.

Change: load the SAME checkpoint onto both GPUs and keep two studies in flight, study i on
GPU i%2. Nothing about a study's own computation changes - same weights, same xw from the
same deterministic eval_windows(), same infer_probs() - so arm_probs[0][i] is bit-for-bit
what the one-GPU run wrote. Only which card computes it changes.

Result placement stays strictly indexed: each in-flight study carries its own index i and
writes arm_probs[0][i], so ordering cannot drift even though completion order can.

Usage:
    python make_solo2g.py <arm_filename.pt> <kaggle-dataset-slug> [K_EVAL]
"""
import ast, json, os, sys

ARM = sys.argv[1]
DATASET = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 48

SRC = "variants/push_solo/knee-mri-raptor-solo-arm.ipynb"
OUT_DIR = "variants/push_solo2g"
SLUG = "knee-mri-raptor-solo-2gpu"
TITLE = "Knee MRI - Raptor Solo Arm on Two GPUs"

# 1. after the models are built, replicate the solo arm onto the second card
SETUP_OLD = ('    _PARALLEL_ARMS = _NG > 1 and len(_sp_models) > 1\n')
SETUP_NEW = ('    _PARALLEL_ARMS = _NG > 1 and len(_sp_models) > 1\n'
             '    # SOLO ON TWO CARDS. One arm means _PARALLEL_ARMS is False and the second GPU sits\n'
             '    # idle for the whole run. Load the SAME checkpoint onto it and alternate studies\n'
             '    # between the cards. Identical weights and identical windows, so every per-study\n'
             '    # result is unchanged; only the card it is computed on differs.\n'
             '    _SOLO_2GPU = _NG > 1 and len(_sp_models) == 1\n'
             '    _sg_pend = {}\n'
             '    if _SOLO_2GPU:\n'
             '        _wp2 = find_weight_file(ARMS[0]["file"])\n'
             '        _m2, _r2 = load_model(_wp2, ARMS[0]["arch"], ARMS[0]["res"], "cuda:1")\n'
             '        assert _r2 == _SP_RES, "replica resolution differs"\n'
             '        _solo_pair = [(_sp_models[0][0], "cuda:0"), (_m2, "cuda:1")]\n'
             '        _solo_ex = _ArmPool2(max_workers=2)\n'
             '        def _solo_infer(_pair, _t):\n'
             '            _mm, _dv = _pair\n'
             '            return infer_probs(_mm, _t, _dv)\n'
             '        print("[gpu] solo arm replicated to cuda:1; two studies in flight", flush=True)\n')

# 2. dispatch each study to the alternating card instead of running it inline
LOOP_OLD = ('                if _PARALLEL_ARMS:\n')
LOOP_NEW = ('                if _SOLO_2GPU:\n'
            '                    _sg_pend[i] = _solo_ex.submit(_solo_infer, _solo_pair[i % 2], xw)\n'
            '                    if len(_sg_pend) >= 2:\n'
            '                        _k = min(_sg_pend)\n'
            '                        arm_probs[0][_k] = _sg_pend.pop(_k).result()\n'
            '                elif _PARALLEL_ARMS:\n')

# 3. drain whatever is still in flight before the arm models are freed
DRAIN_OLD = ('    del _sp_models\n')
DRAIN_NEW = ('    if _SOLO_2GPU:\n'
             '        for _k in sorted(_sg_pend):\n'
             '            arm_probs[0][_k] = _sg_pend.pop(_k).result()\n'
             '        assert not _sg_pend, "studies left in flight"\n'
             '        _solo_ex.shutdown(wait=True)\n'
             '        del _solo_pair, _m2\n'
             '    del _sp_models\n')

MD = """# Knee MRI - Raptor Solo Arm on Two GPUs

The same single CoAtNet-384 model as our solo entry, and the same predictions. The only
change is that the notebook now uses both graphics cards instead of one.

A solo run has one model, and the existing code only runs cards in parallel when there are
two different models, so the second card was idle for the whole run. The one checkpoint is
now loaded onto both cards and studies are alternated between them, with two studies in
flight at a time. Each study is computed from the same weights and the same windows as
before, so the numbers it produces are unchanged; only the card doing the arithmetic differs.

Reading the image files is about three quarters of the run time and is untouched, so this
shortens the run by roughly a tenth rather than halving it.

The harness is derived from Roman Tamrazov's Apache-2.0
[DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4); its
transformer, RadImageNet and calibration stages are removed here, so only the CoAtNet arm runs.
"""

nb = json.load(open(SRC, encoding="utf-8"))
h = {"setup": 0, "loop": 0, "drain": 0}
for c in nb["cells"]:
    if c["cell_type"] == "markdown":
        c["source"] = MD.splitlines(keepends=True)
        continue
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if SETUP_OLD in j:
        j = j.replace(SETUP_OLD, SETUP_NEW, 1); h["setup"] += 1
    if LOOP_OLD in j:
        j = j.replace(LOOP_OLD, LOOP_NEW, 1); h["loop"] += 1
    if DRAIN_OLD in j:
        j = j.replace(DRAIN_OLD, DRAIN_NEW, 1); h["drain"] += 1
    c["source"] = j.splitlines(keepends=True)
assert h == {"setup": 1, "loop": 1, "drain": 1}, "patch sites matched %s" % h

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
assert body.count(ARM) == 1, "solo arm not wired exactly once"
assert "raptor_ft_confdist_swa.pt" not in body and "raptor_ft_coatnet_w80_full_swa.pt" not in body
assert body.count('"w": 1.0') == 1 and '"w": 0.5' not in body, "not a solo weighting"
# the speed wins from the 29-minute kernel must all survive
for marker in ("_hdr_one", "_px_one", "_prep_study", "_PARALLEL_ARMS", "[1pass]"):
    assert marker in body, "lost optimisation: %s" % marker
assert "_A5_SAVED" not in body and "_rad" + "_main" not in body, "a cut stage came back"
assert "K_EVAL = %d" % K in body, "K_EVAL is not %d" % K
# the new path must be complete: replicate, dispatch, drain
for marker in ("_SOLO_2GPU", "_solo_pair", "_sg_pend", "_solo_ex.shutdown"):
    assert marker in body, "incomplete two-GPU path: %s" % marker
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations",
                                               "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("variants/push_solo/kernel-metadata.json", encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
if DATASET not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DATASET)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("built %s" % OUT_DIR)
print("  solo arm %s replicated to cuda:1, two studies in flight" % ARM)
print("  patch sites: %s" % h)
print("  datasets: %s" % meta["dataset_sources"])
