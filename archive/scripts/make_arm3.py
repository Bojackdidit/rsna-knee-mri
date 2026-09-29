#!/usr/bin/env python3
"""Append a THIRD arm to the champion kernel, with a PER-ARM inference crop.

The champion runs two arms and both are the same architecture (coatnet_rmlp_2_rw_384 @384), so our
half is maximally correlated - consistent with five different blend configs all returning 0.940.
The gold-58 OOF says CoAtNet siblings are what add here (+0.0043 to +0.0067 blended onto the live
arm) while EfficientNetV2 adds almost nothing (+0.0004 to +0.0012).

⛔ WHY THE CROP IS PER ARM. A crop-trained checkpoint MUST be served at the same magnification it was
fitted at, or every window it sees is at a different scale from everything it learned on. But
eval_windows is shared by every arm in the loop, so applying a crop globally would also crop the two
UNCROPPED live arms and wreck them. The crop therefore travels in the ARMS entry and is passed per
call; arms without it are byte-identical to today.

No structural change is needed otherwise: ARMS weights are normalised at run time, and load_model
reads `arch` and `res` from the checkpoint with F_dim=bb.num_features.
"""
import ast, argparse, difflib, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="checkpoint filename as it lands in /kaggle/input")
ap.add_argument("--arch", required=True)
ap.add_argument("--res", type=int, required=True)
ap.add_argument("--w", type=float, default=0.25)
ap.add_argument("--crop", type=float, default=1.0,
                help="inference crop for THIS arm only; MUST equal its training --crop_frac")
ap.add_argument("--dataset", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_" + a.tag
SLUG = ("knee-mri-third-arm-" + a.tag).lower()
TITLE = "Knee MRI - Third Arm " + a.tag

OLD_ARMS = '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n]'
NEW_ARMS = (
    '    {"file": "raptor_ft_confdist_swa.pt",              "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
    '    # Third arm. "crop" is PER ARM and must equal the --crop_frac this checkpoint was trained\n'
    '    # with; it cannot be applied globally because eval_windows is shared by every arm.\n'
    '    {"file": "%s", "arch": "%s", "res": %d, "w": %s, "crop": %s},\n]'
    % (a.file, a.arch, a.res, a.w, a.crop))

OLD_SIG = "def eval_windows(vol, mask, k, res, norm=NORM):"
NEW_SIG = "def eval_windows(vol, mask, k, res, norm=NORM, crop=1.0):"

OLD_TRI = ("        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0\n"
           "        t = torch.from_numpy(tri)\n"
           "        if t.shape[-1] != res:")
NEW_TRI = ("        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0\n"
           "        t = torch.from_numpy(tri)\n"
           "        # Exact-centre crop for THIS arm only. No jitter: the trainer jitters +/-4% as\n"
           "        # augmentation, serving must be deterministic. crop=1.0 leaves an arm untouched.\n"
           "        if crop < 1.0:\n"
           "            _H, _W = int(t.shape[-2]), int(t.shape[-1])\n"
           "            _ch, _cw = int(round(_H * crop)), int(round(_W * crop))\n"
           "            _y0, _x0 = (_H - _ch) // 2, (_W - _cw) // 2\n"
           "            t = t[..., _y0:_y0 + _ch, _x0:_x0 + _cw]\n"
           "        if t.shape[-1] != res:")

OLD_CALL = "                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)"
NEW_CALL = ("                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM,\n"
            "                                  crop=float(arm.get('crop', 1.0)))")

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"arms": 0, "sig": 0, "tri": 0, "call": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for key, old, new in (("arms", OLD_ARMS, NEW_ARMS), ("sig", OLD_SIG, NEW_SIG),
                          ("tri", OLD_TRI, NEW_TRI), ("call", OLD_CALL, NEW_CALL)):
        if old in j:
            j = j.replace(old, new, 1)
            hits[key] += 1
    c["source"] = j.splitlines(keepends=True)
assert all(v == 1 for v in hits.values()), "anchors matched %s (each must be exactly 1)" % hits
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
if a.dataset not in meta["dataset_sources"]:
    meta["dataset_sources"].append(a.dataset)
assert meta.get("is_private") is True, "kernel must stay private"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("anchors:", hits, "| all cells parse | private OK")
print("built %s: arm %s (%s @%d, w=%s, crop=%s), dataset %s"
      % (OUT_DIR, a.file, a.arch, a.res, a.w, a.crop, a.dataset))
