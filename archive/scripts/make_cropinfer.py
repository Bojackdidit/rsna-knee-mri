#!/usr/bin/env python3
"""Apply the SAME centre crop at inference that a crop-trained arm was fitted with.

⛔ This is not optional and it is not a tuning knob. If an arm is trained on the middle CROP_FRAC of
each slice but served the full field of view, every window it sees at test time is at a different
magnification from everything it learned on. The adversarial review flagged this as the silent
killer of the whole ROI-crop lever: it would not error, it would just quietly invert the result and
we would wrongly retire the one axis that survived refutation.

Rules:
- EXACT CENTRE at inference. The trainer jitters the crop centre by +/-4% as augmentation; serving
  must not jitter.
- Only ship this together with an arm actually trained at the same --crop_frac. Applying it to the
  existing uncropped arms is the same mismatch in reverse and will cost score.
- Our arm's build_study already crops to a physical size in mm before this point, so this composes
  multiplicatively: 0.70 of a ~140 mm crop is ~98 mm. That is deliberate magnification, but it is
  also why 0.80 exists as a fallback - the champion's own comment warns the collateral ligaments and
  lateral meniscus live in the periphery that an aggressive crop throws away.

Usage: make_cropinfer.py <crop_frac> <arm_file.pt> <arm_arch> <arm_res> <dataset_slug> <tag>
"""
import ast, difflib, json, os, sys

CROP = float(sys.argv[1])
ARM_FILE, ARM_ARCH, ARM_RES = sys.argv[2], sys.argv[3], int(sys.argv[4])
DATASET, TAG = sys.argv[5], sys.argv[6]
SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = f"variants/push_{TAG}"
SLUG = f"knee-mri-{TAG}".lower().replace("_", "-")
TITLE = f"Knee MRI - {TAG.replace('_', ' ').title()}"

OLD = """        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:"""
NEW = f"""        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        # EXACT-CENTRE crop, matching the --crop_frac this arm was trained with. No jitter here:
        # the trainer jitters +/-4% as augmentation, serving must be deterministic and centred.
        _CROP_FRAC = {CROP}
        if _CROP_FRAC < 1.0:
            _H, _W = int(t.shape[-2]), int(t.shape[-1])
            _ch, _cw = int(round(_H * _CROP_FRAC)), int(round(_W * _CROP_FRAC))
            _y0, _x0 = (_H - _ch) // 2, (_W - _cw) // 2
            t = t[..., _y0:_y0 + _ch, _x0:_x0 + _cw]
            if not globals().get('_CROPINFER_LOGGED'):
                globals()['_CROPINFER_LOGGED'] = True
                print(f'CROP-INFER: centre crop {{_CROP_FRAC}} -> {{_ch}}x{{_cw}} from {{_H}}x{{_W}}', flush=True)
        if t.shape[-1] != res:"""

OLD_ARMS = '''ARMS = [
    {"file": "raptor_ft_coatnet_v11blend_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},'''
NEW_ARMS = f'''ARMS = [
    # Crop-trained arm. It MUST be served with the matching exact-centre crop above.
    {{"file": "{ARM_FILE}", "arch": "{ARM_ARCH}", "res": {ARM_RES}, "w": 0.5}},'''

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"crop": 0, "arms": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits["crop"] += 1
    if OLD_ARMS in j:
        j = j.replace(OLD_ARMS, NEW_ARMS, 1); hits["arms"] += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == {"crop": 1, "arms": 1}, hits
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
if DATASET not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DATASET)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print(f"built {OUT_DIR}: crop {CROP} at inference + arm {ARM_FILE} ({ARM_ARCH} @{ARM_RES})")
print(f"  gate string: 'CROP-INFER: centre crop {CROP}'")
