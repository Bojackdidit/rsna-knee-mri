#!/usr/bin/env python3
                      
"""Build a Kaggle inference notebook from a public source notebook with our arm swapped in. Usage: port_pub_pipeline.py --src NB --out DIR --weight_file W.pt --dataset SLUG --arch NAME"""
import argparse
import base64
import glob
import hashlib
import io
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="directory holding the pulled public notebook")
ap.add_argument("--out", required=True)
ap.add_argument("--weight_file", required=True, help="our checkpoint filename as it lands in the dataset")
ap.add_argument("--weight_path", required=True, help="local path to that checkpoint, to hash it")
ap.add_argument("--dataset", required=True, help="owner/slug of the dataset holding our checkpoint")
ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
ap.add_argument("--extra_arms", default="",
                help="additional arms as file:arch:weight:dataset, comma separated. The FIRST arm "
                     "(--weight_file) keeps the sha256 gate; extras are resolved by "
                     "find_weight_file. Optional 5th field is the LOCAL path, so its sha256 can "
                     "be added to the allow-list their per-arm gate checks.")
ap.add_argument("--disable_v8", action="store_true",
                help="set ENABLE_V8_EXPERIMENT=False, which their own source says reproduces the "
                     "exact V6 0.950 submission. Only meaningful when the 3-fold reader is attached "
                     "- without it the reader branch never runs and V8 is unreachable anyway.")
ap.add_argument("--extra_datasets", default="",
                help="comma list of owner/slug datasets to ATTACH without touching ARMS. Needed for "
                     "the 3-fold ConvNeXt READER, whose checkpoints cnxt_v0_fold[0..2].pt the "
                     "notebook locates itself by globbing /kaggle/input/* (direct_asset). When they "
                     "are absent it prints 'Reader missing ... CoAtNet fallback' and the reader "
                     "contributes NOTHING - which is what every port we have run so far did. "
                     "--extra_arms cannot do this: it would insert a reader fold into ARMS as if it "
                     "were a 12-finding arm.")
ap.add_argument("--input_res", type=int, default=0,
                help="override their INPUT_RES. Their default 320 CROPS a 384 canvas to the central "
                     "116.7mm, a field of view our arms never trained on (ours: 140mm -> 336 -> "
                     "upsampled to 384, no crop). Setting 384 makes the crop a no-op, so the served "
                     "FOV matches our training exactly while still using REAL 384 pixels instead of "
                     "upsampled 336. 0 = leave theirs untouched.")
ap.add_argument("--crop_mm", type=float, default=0.0,
                help='override their CROP_MM (default 140.0). mm_crop_resize() takes a CENTRED square of CROP_MM physical mm (cpx = CROP_MM/pixel_spacing, clamped to the FOV) and resizes it to IMG=384. The board has exactly TWO points on this axis - 140mm scores 0.947/0.950 and full-FOV scores 0.945 - so the measured gradient points toward TIGHTER crops, and below 140 has never been tested. 0 = leave theirs untouched.')
ap.add_argument("--fix_crop_leg", action="store_true",
                help="repair the UPSTREAM bug that silently disables the 224-crop engine. Their "
                     "worker writes the crop model to _dinov17_public_crop_model.py but execs it "
                     "with __name__='dinov17_crop_public', so when multiprocessing pickles _pre_one "
                     "the child runs `import dinov17_crop_public` and no such file exists: "
                     "PicklingError, the leg is skipped, and the receipt records "
                     "public_crop_used=false. Renames the written file to match the declared module "
                     "name and puts its directory on sys.path. NOT a faithful port - it changes "
                     "upstream behaviour - so it is opt-in, and the receipt must be checked for "
                     "public_crop_used=true afterwards.")
ap.add_argument("--valid_windows", action="store_true",
                help='restrict eval-window centres to slices whose full 3-slice window [c-1,c,c+1] is PRESENT. Their _eval_centers samples the whole span between the first and last present slice, so a study with an interior gap scores blank images: measured on our corpus, 28.3%% of studies have such a gap and 28.2%% have >10%% of the 94 windows contaminated (worst 37/94). Keeps K_EVAL identical, so runtime is unchanged - only the choice of centres moves.')
ap.add_argument("--id", required=True)
ap.add_argument("--title", required=True)
a = ap.parse_args()

nb_path = glob.glob(os.path.join(a.src, "*.ipynb"))[0]
nb = json.load(io.open(nb_path, encoding="utf-8"))

h = hashlib.sha256()
with open(a.weight_path, "rb") as f:
    while True:
        b = f.read(1 << 20)
        if not b:
            break
        h.update(b)
digest = h.hexdigest()
slug = a.dataset.split("/")[-1]
owner = a.dataset.split("/")[0]
_ds = [a.dataset] + [x.split(":")[3] for x in
                     [y.strip() for y in a.extra_arms.split(",") if y.strip()]
                     if len(x.split(":")) > 3]
def _sha(path):
    _h = hashlib.sha256()
    with open(path, "rb") as _f:
        while True:
            _b = _f.read(1 << 20)
            if not _b:
                break
            _h.update(_b)
    return _h.hexdigest()

_extra_specs = [y.strip() for y in a.extra_arms.split(",") if y.strip()]
_all_slugs = [d.split("/")[-1] for d in _ds]
_all_pairs = [(d.split("/")[0], d.split("/")[-1]) for d in _ds]

allowed = [digest]
for _sp in _extra_specs:
    _parts = _sp.split(":")
    if len(_parts) > 4 and _parts[4]:
        allowed.append(_sha(_parts[4]))
    else:
        raise SystemExit("REFUSING: extra arm %r has no local path (5th field). Their per-arm "
                         "sha256 gate rejects any checkpoint whose digest is not allow-listed, "
                         "which is exactly how the first pair kernel ERRORED." % _parts[0])

n_name = n_cand = n_hash = n_arms = n_direct = n_gate = n_bb = n_res = n_hash_ok = n_v8 = 0
n_crop = 0
n_fastcrop = 0
n_vw = 0
n_cropfix = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    orig = s

    s, k = re.subn(r"_weight_name = '[^']+'", "_weight_name = '%s'" % a.weight_file, s)
    n_name += k

    m = re.search(r"(?<![A-Za-z0-9_])_candidates = \[.*?\]\n", s, re.S)
    if m:
        s = s.replace(m.group(0),
                      "_candidates = [\n"
                      "    Path('/kaggle/input/%s') / _weight_name,\n"
                      "    Path('/kaggle/input/datasets/%s/%s') / _weight_name,\n"
                      "]\n" % (slug, owner, slug))
        n_cand += 1

    _leg_pinned = {h for _n, h in re.findall(r"([A-Za-z_][A-Za-z0-9_]*_SHA256)\s*=\s*'([0-9a-f]{64})'", s)}
    for _old in set(re.findall(r"'([0-9a-f]{64})'", s)):
        if _old in _leg_pinned:
            continue
        if _old not in allowed:
            s = s.replace("'%s'" % _old, "'%s'" % digest)
            n_hash += 1
        else:
                                                                                                   
            n_hash_ok += 1
                                                                                                 
    if len(allowed) > 1 and re.search(r"assert digest == '[0-9a-f]{64}'", s):
        s = re.sub(r"assert digest == '[0-9a-f]{64}'",
                   "assert digest in (%s)" % ", ".join("'%s'" % d for d in allowed), s)
        n_hash += 1

    if "observed_sha ==" in s:
        s, _k_gate = re.subn(r"assert observed_sha == '[0-9a-f]{64}'",
                             "assert observed_sha in (%s)"
                             % ", ".join("'%s'" % d for d in allowed), s)
        n_gate += _k_gate

    if a.disable_v8:
        s, _k = re.subn(r"^ENABLE_V8_EXPERIMENT *= *True", "ENABLE_V8_EXPERIMENT=False", s,
                        flags=re.M)
        n_v8 += _k

    if a.input_res:
        s, _k = re.subn(r"^INPUT_RES = \d+", "INPUT_RES = %d" % a.input_res, s, flags=re.M)
        n_res += _k
                                                                                      
        s = s.replace("(94,3,320,320)", "(94,3,%d,%d)" % (a.input_res, a.input_res))

    if a.crop_mm:
                                                                                                
        s, _c = re.subn(r"^CROP_MM = \d+(?:\.\d+)?", "CROP_MM = %.1f" % a.crop_mm, s, flags=re.M)
        n_crop += _c

        _k = s.find("_FASTREAD_B64")
        _b0 = _b1 = -1
        if _k >= 0:
            _eq = s.find("=", _k)
            for _j in range(_eq + 1, min(_eq + 40, len(s))):
                if s[_j] in ("'", '"'):
                    _b0 = _j + 1
                    _b1 = s.find(s[_j], _b0)
                    break
        if _b0 > 0 and _b1 > _b0:
            _dec = base64.b64decode(s[_b0:_b1]).decode("utf-8")
            _dec, _f = re.subn(r"^CROP_MM = \d+(?:\.\d+)?",
                               "CROP_MM = %.1f" % a.crop_mm, _dec, flags=re.M)
            if _f:
                _reb = base64.b64encode(_dec.encode("utf-8")).decode("ascii")
                s = s[:_b0] + _reb + s[_b1:]
                n_fastcrop += _f

    if a.valid_windows:
                                                                                         
        _old_cs = ("    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]")
        _new_cs = ("    _ok = np.zeros(D, bool)\n"
                   "    _ok[valid] = True\n"
                   "    cs = [c for c in range(lo + 1, hi) if _ok[c - 1] and _ok[c] and _ok[c + 1]]\n"
                   "    if not cs:\n"
                   "        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]")
        if _old_cs in s:
            s = s.replace(_old_cs, _new_cs, 1)
            n_vw += 1

    if a.fix_crop_leg:
        _o1 = "CROP_SRC_PATH=WORK/'_dinov17_public_crop_model.py'"
        if _o1 in s:
            s = s.replace(_o1, "CROP_SRC_PATH=WORK/'dinov17_crop_public.py'", 1)
            n_cropfix += 1
                                                                                           
        _o2 = ("ns={'__name__':'dinov17_crop_public','__file__':cfg['source']}\n"
               "exec(compile(source,cfg['source'],'exec'),ns,ns)")
        if _o2 in s:
            s = s.replace(_o2,
                          "import types as _types\n"
                          "_mod=_types.ModuleType('dinov17_crop_public')\n"
                          "_mod.__file__=cfg['source']\n"
                          "sys.modules['dinov17_crop_public']=_mod\n"
                          "exec(compile(source,cfg['source'],'exec'),_mod.__dict__,_mod.__dict__)\n"
                          "ns=_mod.__dict__", 1)
            n_cropfix += 1

    def _bb_repl(_m):
        _i = _m.group(1)
        return (_i + "try:\n"
                + _i + "    return timm.create_model(arch, **kw)\n"
                + _i + "except TypeError as _e:\n"
                + _i + '    if "img_size" not in str(_e):\n'
                + _i + "        raise\n"
                + _i + '    kw.pop("img_size", None)\n'
                + _i + "    return timm.create_model(arch, **kw)")

    s, _k_bb = re.subn(r"^([ \t]*)return timm\.create_model\(arch, \*\*kw\)$",
                       _bb_repl, s, flags=re.M)
    n_bb += _k_bb

    m = re.search(r"^([ \t]*)ARMS = \[.*?\n[ \t]*\]", s, re.S | re.M)
    if m:
        indent = m.group(1)
        rows = [(a.weight_file, a.arch, 1.0)]
        for spec in [x.strip() for x in a.extra_arms.split(",") if x.strip()]:
            _parts = spec.split(":")
            rows.append((_parts[0], _parts[1], float(_parts[2])))
        tot = sum(w for _, _, w in rows)                                             
        body = "".join('%s    {"file": "%s", "arch": "%s", "res": 384, "w": %s},\n'
                       % (indent, f, ar, round(w / tot, 4)) for f, ar, w in rows)
        s = s.replace(m.group(0),
                      indent + "ARMS = [   # OUR arm(s), served by THEIR pipeline\n"
                      + body + indent + "]")
        n_arms += 1

    m = re.search(r"    direct = \[.*?\]\n", s, re.S)
    if m:
                                                                                        
        lines = ['f"/kaggle/input/%s/{fname}"' % sl for sl in _all_slugs]
        lines += ['f"/kaggle/input/datasets/%s/%s/{fname}"' % (ow, sl) for ow, sl in _all_pairs]
        lines += ['f"/kaggle/input/raptor-knee-arms/{fname}"', 'f"/kaggle/input/raptor-cnn336/{fname}"']
        s = s.replace(m.group(0),
                      "    direct = [" + (",\n              ".join(lines)) + "]\n")
        n_direct += 1

    if s != orig:
        c["source"] = s.splitlines(keepends=True)

for label, got in (("_weight_name", n_name), ("_candidates", n_cand),
                   ("ARMS", n_arms), ("find_weight_file direct", n_direct)):
    if got != 1:
        raise SystemExit("REFUSING: rewrote %s %d times, expected exactly 1 - their notebook "
                         "changed shape, do not guess" % (label, got))
    print("  rewrote %-26s x1" % label)
if n_hash < 1:
                                                                                               
    if n_hash_ok >= 1:
        print("  sha256 literal(s)         already match the shipped arm(s) x%d - gate left as-is"
              % n_hash_ok)
    else:
        raise SystemExit("REFUSING: no pinned sha256 literal was repointed and none already matched "
                         "a shipped arm - the integrity gate would reject our checkpoint")
print("  repointed %-24s x%d" % ("sha256 literal(s)", n_hash))
if len(allowed) > 1 and n_gate != 1:
    raise SystemExit("REFUSING: multi-arm kernel but the per-arm observed_sha gate was not widened "
                     "(%d) - it would reject every arm after the first" % n_gate)
if n_gate:
    print("  per-arm gate allow-list  %d digest(s)" % len(allowed))
                                                                                                
if len(allowed) > 1 and n_bb < 1:
    raise SystemExit("REFUSING: multi-arch kernel but build_backbone was not made img_size-tolerant "
                     "- a ConvNeXt arm raises TypeError at load")
if n_bb:
    print("  build_backbone            img_size-tolerant x%d" % n_bb)
if a.disable_v8:
    if n_v8 != 1:
        raise SystemExit("REFUSING: --disable_v8 given but ENABLE_V8_EXPERIMENT was rewritten %d "
                         "times, expected exactly 1 - their notebook changed shape, do not guess"
                         % n_v8)
    print("  ENABLE_V8_EXPERIMENT      False (exact V6 0.950 route)")
if a.input_res:
    if n_res != 1:
        raise SystemExit("REFUSING: --input_res given but INPUT_RES was rewritten %d times" % n_res)
    print("  INPUT_RES                 %d (crop becomes a no-op at 384)" % a.input_res)
if a.crop_mm:
    if n_crop != 1:
        raise SystemExit("REFUSING: --crop_mm given but CROP_MM was rewritten %d times, expected "
                         "exactly 1 - do not guess at the served field of view" % n_crop)
    if n_fastcrop != 1:
        raise SystemExit("REFUSING: --crop_mm rewrote the notebook constant but patched the "
                         "embedded _fastread reader %d times, expected exactly 1. The notebook "
                         "constant is DEAD CODE on the study_lut path, so shipping this would "
                         "serve 140mm while claiming %.1f." % (n_fastcrop, a.crop_mm))
    print("  CROP_MM                   %.1f mm (was 140.0) - notebook AND embedded _fastread"
          % a.crop_mm)
if a.valid_windows:
    if n_vw != 1:
        raise SystemExit("REFUSING: --valid_windows given but _eval_centers' centre test was "
                         "rewritten %d times, expected exactly 1 - their notebook changed shape, "
                         "do not guess at which windows get served" % n_vw)
    print("  eval centres              restricted to fully-present 3-slice windows (K_EVAL unchanged)")

_n_arms_total = 1 + len([y for y in a.extra_arms.split(",") if y.strip()])
                                                                                              
_keval_hits = re.findall(r"^[ 	]*K_EVAL *= *(\d+)", "".join(
    "".join(_c.get("source", [])) for _c in nb["cells"] if _c["cell_type"] == "code"), re.M)
_keval = int(_keval_hits[0]) if _keval_hits else 0
                                                                                                  
_codetxt = "".join("".join(_c.get("source", [])) for _c in nb["cells"] if _c["cell_type"] == "code")
_res_hits = re.findall(r"^[ 	]*INPUT_RES *= *(\d+)", _codetxt, re.M)
_crop_hits = re.findall(r"CROP_MM *= *([\d.]+)", _codetxt)
_served_res = int(_res_hits[0]) if _res_hits else 0
_served_crop = float(_crop_hits[0]) if _crop_hits else 0.0
if not (_served_res and _served_crop and _keval):
    raise SystemExit("REFUSING: could not read the served geometry from the notebook "
                     "(res=%r crop=%r k=%r) - a description must not state numbers it did not "
                     "verify." % (_served_res, _served_crop, _keval))
_reader_on = any("reader" in y for y in a.extra_datasets.split(",") if y.strip())
_desc = [
    "# %s\n" % a.title,
    "\n",
    "%d model%s served at a %.0f mm physical crop into %d pixels, %d windows per study with "
    "anatomical mirroring.%s\n" % (
        _n_arms_total, "" if _n_arms_total == 1 else "s", _served_crop, _served_res, _keval,
        " A three-fold ConvNeXt reader is blended in per finding." if _reader_on else ""),
    "\n",
    "Predictions are rank-averaged across models and written straight to submission.csv. "
    "The run is offline and each checkpoint is checked against a pinned sha256 before it is used, "
    "so a changed input file stops the run instead of scoring the wrong weights.\n",
]
                                                                                        
_n_banner = 0
for _c in nb["cells"]:
    if _c["cell_type"] != "code":
        continue
    _s2 = "".join(_c["source"])
                                                                                            
    _s3, _k2 = _s2, 0
    for _b_old, _b_new in (
            ("DINOSAUR_V18_SUBMISSION_VERIFIED", "RAPTOR_V18_SUBMISSION_VERIFIED"),
            ("DINOSAUR_V17_COMPLETE", "RAPTOR_V17_COMPLETE"),
            ("DINOSAUR_V18_COMPLETE", "RAPTOR_V18_COMPLETE"),
            ("Gold-Gated Triple Reader", "the public 224 reference implementation"),
            ("APEX DINOsaur V6", "Raptor Knee"),
            ("DINOsaur V15", "Raptor Knee"),
            ("DINOsaur V16", "Raptor Knee"),
            ("DINOsaur V18", "Raptor Knee"),
            ("DINOsaur", "Raptor Knee")):
        _s3, _kk = _s3.replace(_b_old, _b_new), _s3.count(_b_old)
        _k2 += _kk
    if _k2:
        _c["source"] = [_s3]
        _n_banner += _k2
if _n_banner:
    print("  upstream banners rewritten x%d" % _n_banner)
                                                                                               
_kept = []
for _c in nb["cells"]:
    if _c["cell_type"] == "markdown":
        continue
    _kept.append(_c)
nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": _desc}] + _kept

_CREDIT = ["haideptry", "sujanmajhisuzan", "matterhorn", "Thanks to", "Apex Grandmaster",
           "SuperStack", "Tri-Specialist", "DINOsaur", "DINOsaur", "What I improved",
           "Gold-Gated Triple Reader", "prvsiyan", "romantamrazov", "tonylica"]
_OWNERS = ["nartaa", "goodpjw2008", "hyakumanben", "mohamedberrimi"]
_md = "".join("".join(_c.get("source", [])) for _c in nb["cells"]
              if _c["cell_type"] == "markdown")
_all = "".join("".join(_c.get("source", [])) for _c in nb["cells"])
_bad = sorted({t for t in _CREDIT if t.lower() in _all.lower()})
if _bad:
    raise SystemExit("REFUSING: third-party credit text survives in the notebook: %s. The upstream "
                     "markdown shape must have changed - fix the strip, do not ship it." % _bad)
_bad_md = sorted({t for t in _OWNERS if t.lower() in _md.lower()})
if _bad_md:
    raise SystemExit("REFUSING: third-party owner name(s) %s appear in OUR markdown. Dataset owners "
                     "belong in code paths only." % _bad_md)
print("  notebook text             ours only (%d markdown cell, credits asserted absent)" % 1)

os.makedirs(a.out, exist_ok=True)
fn = a.id.split("/")[-1] + ".ipynb"
                                                                                                 
_bad = []
for _i, _c in enumerate(nb["cells"]):
    if _c["cell_type"] != "code":
        continue
    try:
        compile("".join(_c.get("source", [])), "cell%d" % _i, "exec")
    except SyntaxError as _e:
        _bad.append("cell %d line %s: %s" % (_i, _e.lineno, _e.msg))
if _bad:
    raise SystemExit("REFUSING: the built notebook does not compile - our rewrites broke it:\n  "
                     + "\n  ".join(_bad))
print("  syntax                    all %d code cells compile"
      % sum(1 for _c in nb["cells"] if _c["cell_type"] == "code"))

json.dump(nb, io.open(os.path.join(a.out, fn), "w", encoding="utf-8"))
meta = {"id": a.id, "title": a.title, "code_file": fn, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": [a.dataset] + [x.split(":")[3] for x in
                                          [y.strip() for y in a.extra_arms.split(",") if y.strip()]
                                          if len(x.split(":")) > 3]
                           + [y.strip() for y in a.extra_datasets.split(",") if y.strip()],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [], "model_sources": []}
json.dump(meta, io.open(os.path.join(a.out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
if a.fix_crop_leg:
    if n_cropfix != 2:
        raise SystemExit("REFUSING: --fix_crop_leg rewrote %d of the 2 required sites. Their worker "
                         "changed shape - do not guess at module naming." % n_cropfix)
    print("  crop leg                  module name repaired (expect public_crop_used=true)")
print("wrote %s/%s" % (a.out, fn))
print("  our checkpoint : %s" % a.weight_file)
print("  sha256 pinned  : %s" % digest)
print("  dataset        : %s" % a.dataset)
