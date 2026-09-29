#!/usr/bin/env python3
"""Arm-only CoAtNets + the CHAMP outside family (5 DINOv2 folds, ALL TWELVE findings).

Why champ rather than the legacy family: legacy carries weight on only FOUR findings by design, so
it can never be a complete outside half. champ's fingerprint declares all twelve targets, is trained
from RULE-based labels (ours are LLM-distilled, so it disagrees by construction), and ships in our
kernel's NATIVE member format - model + fingerprint + fold - so the existing machinery can run it.

    champ fingerprint: img 224, crop_mm 130.0, group 3 x n_group 3 = 9 slices, window 0.35-0.65,
                       backbone dinov2, targets all 12, folds 5, targets_from "rules"

Weight is flat across every target, so it is an opinion alongside our arms rather than a takeover of
a few findings - the shape that killed OrthoDiffusion.
"""
import io, json, os, sys, ast

SRC = "variants/push_fastdcm/knee-mri-raptor-direct-dicom-read.ipynb"
OUT = "variants/push_champ_arm"
SLUG = "knee-mri-champ-outside-family"
TITLE = "Knee MRI Champ Outside Family"
CHAMP_W = float(os.environ.get("CHAMP_W", "6.0"))

LOADER = '''def find_champ_members():
    base = Path('/kaggle/input')
    found = []
    if not base.is_dir():
        return found
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        for fn in sorted(files):
            if re.fullmatch(r'champ_fold[0-9]+[.]pt', fn):
                found.append(Path(root) / fn)
    return sorted(found)


CHAMP_MEMBER_WEIGHT = __CW__


def legacy_group_members():
    # The CHAMP family, loaded through the legacy hook. Unlike the legacy bundles it declares ALL
    # TWELVE targets in its own fingerprint, so it can carry the whole outside half instead of four
    # findings. Its labels come from rules rather than our LLM distillation, so it disagrees with our
    # arms by construction - the only kind of member that has ever paid on this competition.
    paths = find_champ_members()
    if not paths:
        log('no champ members attached; outside half skipped')
        return {}
    out_ms, key = ([], None)
    for p in paths:
      try:
        b = torch.load(p, map_location='cpu', weights_only=False)
        fp = b.get('fingerprint') or {}
        if list(fp.get('targets', TARGETS)) != TARGETS:
            log('champ ' + p.name + ': target contract differs; skipped')
            continue
        gr, n_gr = (int(fp.get('group', 3)), int(fp.get('n_group', 3)))
        win = str(fp.get('window', '0.35,0.65')).split(',')
        band = [float(win[0]), float(win[1])]
        key = json.dumps({'img': int(fp.get('img', 224)), 'group': gr, 'slices': gr * n_gr,
                          'crop_mm': float(fp.get('crop_mm', 130.0)), 'band': band,
                          'rules': RULES_LEGACY, 'slots': [s[0] for s in SLOTS]}, sort_keys=True)
        fold = int(b.get('fold', 0))
        # champ names its tensors encoder.module.* ; our Model wants backbone.* . Same
        # architecture (dinov2 small, 233 tensors), different prefix - remap rather than
        # rebuild. 'mean'/'std' are normalisation buffers our Model does not carry.
        # Only the prefix differs: champ names tensors encoder.module.*, our Model wants
        # backbone.* . KEEP mean/std - our Model carries them as buffers. champ has no
        # head.slot_prior because it was not trained with the slot prior, so prior=False below.
        _sd = {}
        for _k, _v in b['model'].items():
            _sd[_k.replace('encoder.module.', 'backbone.', 1) if _k.startswith('encoder.module.') else _k] = _v
        out_ms.append({'id': 'champ-f' + str(fold), 'fold': fold, 'state': _sd,
                       'holdout': None, 'weight': CHAMP_MEMBER_WEIGHT,
                       'target_weight': [CHAMP_MEMBER_WEIGHT for _t in TARGETS],
                       'pixel_group': key,
                       'config': {'unfreeze_last': 6, 'variant': 'small',
                                  'pool': 'cls_mean_focal', 'prior': False}})
        log('champ ' + p.name + ': fold ' + str(fold) + ' joined')
      except Exception as exc:
        log('champ ' + p.name + ' unusable (' + type(exc).__name__ + '); skipped')
    if out_ms:
        log('champ family: ' + str(len(out_ms)) + ' member(s), flat weight on all 12 targets')
    return {key: out_ms} if out_ms and key else {}
'''.replace("__CW__", repr(CHAMP_W))

nb = json.load(io.open(SRC, encoding="utf-8"))
STUB = "def legacy_group_members():\n    return {}\n"
OLD_MAN = ("    man = json.loads((Path(path) / 'manifest.json').read_text())\n"
           "    members = man['members']\n")
NEW_MAN = ("    # partner half is dead weight: arm-only scored 0.941 vs the blend's 0.940 at 1/6 compute\n"
           "    members = []\n")
OLD_SEED = ("print('ONEPASS: partner half not run', flush=True)\n"
            "import pandas as _ao_pd\n"
            "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')\n"
            "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)\n")
NEW_SEED = ("print('ONEPASS: partner NOT run; CHAMP outside family IS run', flush=True)\n"
            "import pandas as _ao_pd\n"
            "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')\n"
            "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "try:\n"
            "    infer_from_package(ASSET / 'rsna-knee-weights', DEVS[0])\n"
            "    print('CHAMP: outside family completed', flush=True)\n"
            "except Exception as _e:\n"
            "    print('CHAMP: post-step raised (' + type(_e).__name__ + ')', flush=True)\n")
OLD_WRITE = ('    sub.to_csv("/kaggle/working/submission.csv", index=False)\n'
             '    print("ONEPASS: wrote submission.csv", flush=True)\n')
NEW_WRITE = ('    _outside = "/kaggle/working/submission.csv"\n'
             '    try:\n'
             '        _prev = pd.read_csv(_outside); _keep = float(_prev[LAB].std().max()) > 1e-9\n'
             '    except Exception:\n'
             '        _keep = False\n'
             '    if _keep:\n'
             '        print("ONEPASS: keeping the outside half already in submission.csv", flush=True)\n'
             '    else:\n'
             '        sub.to_csv(_outside, index=False)\n'
             '        print("ONEPASS: wrote submission.csv (outside half was constant)", flush=True)\n')

h = {"stub": 0, "man": 0, "seed": 0, "write": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    o = s
    if STUB in s:
        s = s.replace(STUB, LOADER, 1)
        h["stub"] += 1
    if OLD_MAN in s:
        s = s.replace(OLD_MAN, NEW_MAN, 1)
        h["man"] += 1
    if OLD_SEED in s:
        s = s.replace(OLD_SEED, NEW_SEED, 1)
        h["seed"] += 1
    if OLD_WRITE in s:
        s = s.replace(OLD_WRITE, NEW_WRITE, 1)
        h["write"] += 1
    if s != o:
        c["source"] = s.splitlines(keepends=True)

print("patch hits:", h)
if any(v != 1 for v in h.values()):
    sys.exit("anchors did not all match exactly once")

os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, SLUG + ".ipynb"), "w", encoding="utf-8"), ensure_ascii=False)
meta = json.load(io.open("variants/push_fastdcm/kernel-metadata.json", encoding="utf-8"))
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
meta.pop("id_no", None)
ds = list(meta.get("dataset_sources", []))
if "stevenleehans/rsna-knee-champ-members-only" not in ds:
    ds.append("stevenleehans/rsna-knee-champ-members-only")
meta["dataset_sources"] = ds
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)

chk = "".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
for n, cond in [("champ loader", "def find_champ_members" in chk),
                ("all-12 weights", "CHAMP_MEMBER_WEIGHT for _t in TARGETS" in chk),
                ("partner dropped", "    members = []" in chk),
                ("champ invoked", "CHAMP: outside family completed" in chk),
                ("no clobber", "keeping the outside half already in submission.csv" in chk),
                ("fastdcm intact", "m, a = _fd_read_one(f)" in chk),
                ("arms intact", "raptor_ft_coatnet_w80_full_swa.pt" in chk)]:
    print("  %-18s %s" % (n, "OK" if cond else "FAIL"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("  all cells parse    OK")
print("title slugifies to id:", TITLE.lower().replace(" ", "-") == SLUG)
