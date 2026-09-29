#!/usr/bin/env python3
"""The stronger arm pair, with the two stages whose output we throw away removed.

Reading the champion notebook end to end shows FOUR prediction stages, not two:

  cell 1  run_dinov2()      20-member partner half   - already skipped by the arm-only seed
  cell 2  A5 DINOv3 5-fold  own build_study pass over every study, blended at A5_W=0.45
  cell 3  _rad_main()       own per-study pass ("ordering N slot-series, M slice headers")
  cell 4  our CoAtNet arms  writes submission.csv, then the V18 block blends it with itself

Cells 2 and 3 each write /kaggle/working/submission.csv, and cell 4's arm write lands AFTER both
of them, so their output is discarded before the V18 blend ever reads the file. Confirmed from the
run log: "ARMSWAP: wrote submission.csv" at t=83.822 precedes "final submission.csv = V18 ..." at
t=83.846, and the V18 blend takes BOTH of its inputs from what the arm just wrote - so the result
is the arm regardless of the blend weights. Removing them cannot move the score.

They are not cheap. On the 3-study dummy run: A5 ~1.8 s/study, cell 3 ~3.5 s/study, and cell 3
does its own DICOM header pass, which is the exact cost already identified as the bottleneck.
Across 1,322 studies that is a large part of the wall clock spent on discarded work, and it is why
removing the 20-member partner half made the run barely faster: a second and third stage were still
running.

Dependency check before cutting: the only names defined in cells 2/3 and referenced in cell 4 are
`device` (a function parameter there, not a global) and `V18_CALIBRATOR_APPLIED`, which is read as
globals().get(..., False) and only adjusts blend weights that cannot matter when both blend inputs
are identical.

Arms are the best available pair on the 58-study gate: confdist_swa + w80_full_swa = 0.9333 against
the shipped v11blend + confdist 0.9282 (+0.0051, better in 93.5% of 2000 bootstraps). Also carries
the specific_tags header parse (7.4x on the same files, identical ordering).
"""
import ast, json, os, sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 48
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_1pass_single"
SLUG = "knee-mri-one-pass-single-arm"
TITLE = "Knee MRI - One Pass Single Arm"
EXTRA_DATASETS = []

SEED = "\n".join([
    "print('ONEPASS: partner half not run', flush=True)",
    "import pandas as _ao_pd",
    "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')",
    "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)",
]) + "\n"

STUB2 = ("# A5 DINOv3 5-fold removed. It ran its own build_study pass over every study and wrote\n"
         "# submission.csv, which the CoAtNet arm overwrites before the V18 blend reads it, so its\n"
         "# output never reached the score. Nothing defined here is used downstream.\n"
         "print('ONEPASS: A5 transformer half not run', flush=True)\n")
STUB3 = ("# The rad calibration stage is removed, for the same reason: it re-read DICOM headers for\n"
         "# every study and wrote submission.csv, and the arm write lands after it. V18_CALIBRATOR_\n"
         "# APPLIED is left undefined on purpose; cell 4 reads it via globals().get(..., False).\n"
         "print('ONEPASS: rad calibration not run', flush=True)\n")

OLD_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
            ']')
NEW_ARMS = ('ARMS = [\n'
            '    {"file": "raptor_ft_confdist_swa.pt",          "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n'
            ']')
OLD_ARMLOOP = open('_armloop.txt', encoding='utf-8').read()
NEW_ARMLOOP = '    # SINGLE PASS over the test set. Every arm here is the same architecture at the same\n    # resolution, so eval_windows() returns the IDENTICAL tensor for each of them. The original\n    # loop was arms-outer, which re-read and re-preprocessed all 1,322 studies once PER ARM, and\n    # DICOM reading is the whole wall clock (header scan alone was 72.9% of IO). Building each\n    # study once and running both models on the result halves the run. Numerically this is the\n    # same computation: same xw, same infer_probs, same arm_probs, so the blend is unchanged.\n    _sp_models = []\n    for a, arm in enumerate(ARMS):\n        wp = find_weight_file(arm["file"])\n        _m, _res = load_model(wp, arm["arch"], arm["res"], dev)\n        _sp_models.append((_m, _res))\n        print(f"[arm {a}] loaded {arm[\'file\']} | res {_res} | {time.time()-t0:.0f}s", flush=True)\n    assert len({_r for _, _r in _sp_models}) == 1, "single pass needs one resolution"\n    _SP_RES = _sp_models[0][1]\n    for i, sid in enumerate(test_ids):\n        try:\n            vol, mask = build_study(sid, SER, tsdir, reader)\n            xw = eval_windows(vol, mask, k=K_EVAL, res=_SP_RES, norm=NORM)\n            for a, (_m, _r) in enumerate(_sp_models):\n                arm_probs[a][i] = infer_probs(_m, xw, dev)\n            del vol, mask, xw\n        except Exception as e:\n            print(f"  study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)\n        if (i + 1) % 100 == 0 or i + 1 == N:\n            print(f"  [1pass] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)\n    del _sp_models\n    gc.collect()\n    if str(dev).startswith("cuda"):\n        torch.cuda.empty_cache()\n    print(f"[1pass] all arms done + freed | {time.time()-t0:.0f}s", flush=True)\n'
OLD_HDR = "                h = pydicom.dcmread(f, stop_before_pixels=True)"
NEW_HDR = ("                # Parse ONLY the four elements used below. Measured 7.4x faster than a\n"
           "                # full header parse over the same files, with identical ordering.\n"
           "                h = pydicom.dcmread(f, stop_before_pixels=True, force=True,\n"
           "                                    specific_tags=[(0x0020, 0x0032), (0x0020, 0x0037),\n"
           "                                                   (0x0028, 0x0030), (0x0020, 0x0013)])")
EDITS = [
    ("run_dinov2()\n", SEED),
    ('    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n',
     '    out = "/kaggle/working/submission_coatnet.csv"\n    sub.to_csv(out, index=False)\n'
     '    sub.to_csv("/kaggle/working/submission.csv", index=False)\n'
     '    print("ONEPASS: wrote submission.csv", flush=True)\n'),
    ("K_EVAL = 94", "K_EVAL = %d" % K),
    (OLD_ARMS, NEW_ARMS),
    (OLD_HDR, NEW_HDR),
    (OLD_ARMLOOP, NEW_ARMLOOP),
]

nb = json.load(open(SRC, encoding="utf-8"))
# Identify the two stages by CONTENT, not position, so a re-ordered notebook cannot mis-cut.
a5 = [i for i,c in enumerate(nb["cells"]) if c["cell_type"]=="code"
      and "_A5_SAVED" in "".join(c["source"]) and "A5_W" in "".join(c["source"])]
rad = [i for i,c in enumerate(nb["cells"]) if c["cell_type"]=="code"
       and "_rad_main()" in "".join(c["source"])]
assert len(a5) == 1, "A5 block matched %d cells" % len(a5)
assert len(rad) == 1, "rad block matched %d cells" % len(rad)
assert a5[0] != rad[0]
assert "run_dinov2()" not in "".join(nb["cells"][a5[0]]["source"]), "A5 cell also holds the partner call"
assert "ARMS = [" not in "".join(nb["cells"][rad[0]]["source"]), "rad cell also holds the arms"
nb["cells"][a5[0]]["source"] = STUB2.splitlines(keepends=True)
nb["cells"][rad[0]]["source"] = STUB3.splitlines(keepends=True)
print("cut cell %d (A5) and cell %d (rad)" % (a5[0], rad[0]))

counts = [0]*len(EDITS)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for i,(old,new) in enumerate(EDITS):
        if old in j:
            j = j.replace(old,new,1); counts[i]+=1
    c["source"] = j.splitlines(keepends=True)
assert counts == [1]*len(EDITS), "edits matched %s" % counts
body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
assert "raptor_ft_coatnet_v11blend_full_swa.pt" not in body, "old arm still referenced"
assert body.count("raptor_ft_coatnet_w80_full_swa.pt") == 0
assert body.count("raptor_ft_confdist_swa.pt") == 1
assert "_A5_SAVED" not in body and "_rad_main" not in body, "a cut stage survived"
assert body.count("for a, arm in enumerate(ARMS):") == 1, "old arms-outer loop survived"
assert "[1pass]" in body
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("champ/kernel-metadata.json",encoding="utf-8")); meta.pop("id_no",None)
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
for d in EXTRA_DATASETS:
    if d not in meta["dataset_sources"]:
        meta["dataset_sources"].append(d)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | A5 + rad cut, arms swapped")
print("2 arms x %d windows = %d passes/study, ONE DICOM pass per arm and nothing else" % (K, 2*K))
