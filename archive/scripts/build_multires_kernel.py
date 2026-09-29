#!/usr/bin/env python3
"""Lift the single-resolution assert so arms we ALREADY OWN can ride together. Zero GPU.

WHY
---
The kernel builds each study's windows ONCE and asserts "single pass needs one resolution", and the
window mask is shared too. So an arm can only ride if it matches 384 AND uses full slots. That is
why the measured arm-count gain (2 arms 0.9331 -> 3 -> 4 arms 0.9356) was written off as an
illusion: coronal48_78_swa is slot-restricted, effv2l480 is 480px, cnv2b336 is 336px, and none of
them could be submitted alongside the champion.

But that assert is an IMPLEMENTATION CHOICE, not a constraint. Building one window tensor per
distinct (resolution, slot-set) costs one extra CPU resize per study - microseconds against the
~14 ms of DICOM work already being done per study - and nothing on the GPU. Arms that share a group
still share a tensor, so the champion pair costs exactly what it costs today.

This unlocks a genuinely multi-FAMILY blend from weights already on disk and already paid for:
CoAtNet-384 x2 + ConvNeXt-Base-384 + ViT-B/16-384 + EfficientNetV2-S-336. Per the gold-58
per-finding profiles these are complementary exactly where we are weak - ConvNeXt carries Synovitis
(0.871 vs the champion's 0.832) and Lateral OA (0.838 vs 0.818), while the champion carries MCL
(0.995 vs 0.930) and Lateral Meniscus (0.878 vs 0.812).

RUNTIME
-------
Arms are placed round-robin across the GPUs and run concurrently, so wall clock is the SLOWEST
card, not the sum. Measured per-study cost at 384/48 windows on an H100: CoAtNet 88.3 ms,
ViT-B/16 22.6, ConvNeXt-Base 47.6, EffV2-S 63.5. Put the two CHEAPEST on the same card.
Do NOT price this with a 3-study dry run - dry runs are warm-up dominated and got ConvNeXt wrong
by 2.5x in the wrong direction.

Usage:
    python build_multires_kernel.py --out variants
"""
import argparse, copy, json, os

BASE_NB = os.path.join("variants", "push_fastdcm_clean",
                       "knee-mri-raptor-direct-dicom-read.ipynb")
BASE_META = os.path.join("variants", "push_fastdcm_clean", "kernel-metadata.json")

# (weights file, arch, res, slot subset or None for all five, blend weight)
# Slot-restricted arms are now legal: they simply form their own window group.
ARM_LIST = [
    ("raptor_ft_confdist_swa.pt",         "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384, None, 1.0),
    ("raptor_ft_coatnet_w80_full_swa.pt", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384, None, 1.0),
    ("raptor_ft_raptor384_cnb.pt",        "convnext_base.fb_in22k_ft_in1k_384",     384, None, 0.7),
    ("raptor_ft_raptor384_vit.pt",        "vit_base_patch16_384.augreg_in21k_ft_in1k", 384, None, 0.5),
    ("raptor_ft_E1b_v2s336_3ser.pt",      "tf_efficientnetv2_s.in21k_ft_in1k",      336, (0, 2, 4), 0.5),
]

OLD_ARMS_START = 'ARMS = [\n    {"file": "raptor_ft_confdist_swa.pt",'
OLD_ARMS_END = '"res": 384, "w": 0.5},\n]'


def arms_literal():
    rows = []
    for f, arch, res, slots, w in ARM_LIST:
        rows.append('    {"file": %r, "arch": %r, "res": %d, "slots": %r, "w": %.2f},'
                    % (f, arch, res, slots, w))
    return "ARMS = [\n" + "\n".join(rows) + "\n]"


OLD_SPMODELS = '''    _sp_models = []
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        _m, _res = load_model(wp, arm["arch"], arm["res"], _arm_devs[a])
        _sp_models.append((_m, _res))
        print(f"[arm {a}] loaded {arm['file']} | res {_res} | dev {_arm_devs[a]} | {time.time()-t0:.0f}s", flush=True)
    assert len({_r for _, _r in _sp_models}) == 1, "single pass needs one resolution"
    _SP_RES = _sp_models[0][1]'''

NEW_SPMODELS = '''    _sp_models = []
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        _m, _res = load_model(wp, arm["arch"], arm["res"], _arm_devs[a])
        _sp_models.append((_m, _res))
        print(f"[arm {a}] loaded {arm['file']} | res {_res} | slots {arm.get('slots')} | "
              f"dev {_arm_devs[a]} | {time.time()-t0:.0f}s", flush=True)
    # MULTI-RESOLUTION. The old code asserted one resolution for every arm, which locked out every
    # non-384 and every slot-restricted arm we ever trained. Instead, group arms by the window
    # tensor they need - (resolution, slot subset) - and build one tensor per distinct group.
    # Arms in the same group still share a tensor, so the champion pair costs exactly what it did.
    # The extra cost is one CPU resize per extra group per study, against ~14 ms of DICOM work.
    _GRP = []                                  # list of (res, slots_tuple_or_None)
    _arm_grp = []                              # arm index -> group index
    for a, arm in enumerate(ARMS):
        _key = (int(_sp_models[a][1]), tuple(arm["slots"]) if arm.get("slots") else None)
        if _key not in _GRP:
            _GRP.append(_key)
        _arm_grp.append(_GRP.index(_key))
    print(f"[groups] {len(_GRP)} window group(s) for {len(ARMS)} arms: {_GRP}", flush=True)
    print(f"[groups] arm -> group {_arm_grp}", flush=True)
    _SP_RES = _sp_models[0][1]'''

OLD_PREP = '''    def _prep_study(sid):
        vol, mask = build_study(sid, SER, tsdir, reader)
        xw = eval_windows(vol, mask, k=K_EVAL, res=_SP_RES, norm=NORM)
        del vol, mask
        return xw'''

NEW_PREP = '''    def _prep_study(sid):
        vol, mask = build_study(sid, SER, tsdir, reader)
        # One window tensor per distinct (resolution, slot subset). Built from the SAME decoded
        # volume, so the expensive part - reading and decoding the DICOMs - happens once.
        _out = []
        for _res, _slots in _GRP:
            _m = mask
            if _slots is not None:
                _keep = np.zeros(len(mask), bool)
                _acc = 0
                for _si, _b in enumerate(SLOT_BUDGETS):
                    if _si in _slots:
                        _keep[_acc:min(_acc + _b, len(mask))] = True
                    _acc += _b
                _m = (np.asarray(mask) > 0) & _keep
                if _m.sum() < 3:
                    _m = mask
            _out.append(eval_windows(vol, _m, k=K_EVAL, res=_res, norm=NORM))
        del vol, mask
        return _out'''

OLD_CONSUME = '''                xw = _fut.pop(i).result()
                if _PARALLEL_ARMS:
                    # Both arms at once, each on its own device. infer_probs moves the tensor.
                    def _run_arm(_t):
                        _a, (_mm, _rr) = _t
                        return _a, infer_probs(_mm, xw, _arm_devs[_a])
                    for _a, _o in _arm_ex.map(_run_arm, list(enumerate(_sp_models))):
                        arm_probs[_a][i] = _o
                else:
                    for a, (_m, _r) in enumerate(_sp_models):
                        arm_probs[a][i] = infer_probs(_m, xw, _arm_devs[a])
                del xw'''

NEW_CONSUME = '''                xws = _fut.pop(i).result()
                if _PARALLEL_ARMS:
                    # All arms at once, round-robin across the GPUs. Each takes the window tensor
                    # its own group produced. Wall clock is the slowest CARD, not the sum.
                    def _run_arm(_t):
                        _a, (_mm, _rr) = _t
                        return _a, infer_probs(_mm, xws[_arm_grp[_a]], _arm_devs[_a])
                    for _a, _o in _arm_ex.map(_run_arm, list(enumerate(_sp_models))):
                        arm_probs[_a][i] = _o
                else:
                    for a, (_m, _r) in enumerate(_sp_models):
                        arm_probs[a][i] = infer_probs(_m, xws[_arm_grp[a]], _arm_devs[a])
                del xws'''

OLD_POOL = '    _arm_ex = _ArmPool2(max_workers=len(_sp_models)) if _PARALLEL_ARMS else None'
NEW_POOL = ('    _arm_ex = _ArmPool2(max_workers=len(_sp_models)) if _PARALLEL_ARMS else None\n'
            '    # With >2 arms a card carries more than one; the pool must be wide enough to keep\n'
            '    # every arm in flight or the "concurrent" placement silently serialises.')

GLOBALS_ANCHOR = '''_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)'''

GLOBALS_NEW = '''_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SLOT_BUDGETS = [s[2] for s in SLOTS]               # 27/21/18/12/18
assert sum(SLOT_BUDGETS) == MAXS, "slot budgets do not sum to MAXS"'''


def patch(nb, old, new, label):
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if old in s:
            hits += s.count(old)
            c["source"] = s.replace(old, new).splitlines(keepends=True)
    if hits != 1:
        raise SystemExit("patch %r matched %d times, expected 1" % (label, hits))


def bad_cells(nb, where):
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        try:
            compile("".join(c["source"]), "%s#%d" % (where, i), "exec")
        except SyntaxError:
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="variants")
    a = ap.parse_args()
    base = json.load(open(BASE_NB, encoding="utf-8"))
    base_bad = bad_cells(base, BASE_NB)

    nb = copy.deepcopy(base)
    # replace the whole ARMS block
    src = "".join(nb["cells"][5]["source"]) if len(nb["cells"]) > 5 else ""
    whole = None
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        i = s.find(OLD_ARMS_START)
        if i >= 0:
            j = s.find(OLD_ARMS_END, i)
            assert j > 0, "could not find the end of the ARMS block"
            whole = s[i:j + len(OLD_ARMS_END)]
            break
    assert whole, "ARMS block not found"
    patch(nb, whole, arms_literal(), "ARMS")
    patch(nb, GLOBALS_ANCHOR, GLOBALS_NEW, "globals")
    patch(nb, OLD_SPMODELS, NEW_SPMODELS, "sp_models")
    patch(nb, OLD_PREP, NEW_PREP, "_prep_study")
    patch(nb, OLD_CONSUME, NEW_CONSUME, "consumer")
    patch(nb, OLD_POOL, NEW_POOL, "arm pool note")

    new_bad = sorted(set(bad_cells(nb, "multires")) - set(base_bad))
    if new_bad:
        raise SystemExit("patch introduced syntax errors in cell(s) %s" % new_bad)

    slug = "knee-mri-multi-resolution-blend"
    d = os.path.join(a.out, "push_" + slug.replace("-", "_"))
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, slug + ".ipynb"), "w", encoding="utf-8"))
    m = dict(json.load(open(BASE_META, encoding="utf-8")))
    m.pop("id_no", None)
    m["id"] = "dreaddevelopment/" + slug
    m["title"] = "Knee MRI - Multi-Resolution Blend"
    m["code_file"] = slug + ".ipynb"
    m["kernel_type"] = "notebook"
    json.dump(m, open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)

    full = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "single pass needs one resolution" not in full, "the assert survived"
    for must in ("_GRP", "_arm_grp", "SLOT_BUDGETS", "xws[_arm_grp["):
        assert must in full, "missing " + must
    grp = sorted({(r, tuple(s) if s else None) for _, _, r, s, _ in ARM_LIST})
    print("[ok] %s -> %s" % (slug, d))
    print("     %d arms, %d window group(s): %s" % (len(ARM_LIST), len(grp), grp))
    print("     WEIGHTS NEEDED in the attached arm dataset:")
    for f, arch, res, s, w in ARM_LIST:
        print("       %-36s %-42s res %d slots %-9s w %.2f" % (f, arch, res, str(s), w))
    print("\n     Attach every one of these before pushing, or find_weight_file raises at load.")


if __name__ == "__main__":
    main()
