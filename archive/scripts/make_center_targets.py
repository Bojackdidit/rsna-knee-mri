#!/usr/bin/env python3
"""Centre-restrict window pooling for the CENTRAL structures only.

The blanket centre-window sweep (make_center.py) cuts every finding's windows the same way. That
is the right first read of the curve, but anatomically it is blunt: Medial OA, Lateral OA, Medial
Meniscus and Lateral Meniscus are LATERALISED by definition and live at the edges of the sagittal
stack, so cutting to the centre should hurt them even if it helps elsewhere.

This variant applies the restriction only where the anatomy is central:
    ACL, MCL            top2         -> centre_top2
    Effusion, PF OA     mean         -> centre_mean
    Synovitis           original_mean-> centre_original_mean
and leaves the lateralised findings (Medial/Lateral OA, Medial/Lateral Meniscus) and the focal ones
that pool with max (Fracture, Contusion, Baker's) reading the full window range exactly as the
champion does.

The kernel already pools per target and already receives the full window axis at that point
(`probs` is [n_windows, n_studies, n_targets]), so this needs no change to decode, to the models,
to the cache or to the blend weight - only to how the existing window axis is reduced per finding.
"""
import ast, difflib, json, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
K = 4
OUT_DIR = "variants/push_center_targets"
SLUG = "knee-mri-centre-windows-central-findings"
TITLE = "Knee MRI - Centre Windows For Central Findings"

OLD_FN = """def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
"""
NEW_FN = f"""# Windows kept when a target's pooling mode is prefixed 'centre_'. The lesion for these
# findings sits centrally in the sagittal stack, so the outer positions mostly add background.
CENTRE_TARGET_K = {K}

def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        probs_t, logits_t, original_t = probs, logits, original_probs
        if mode.startswith('centre_'):
            mode = mode[len('centre_'):]
            n_win = probs.shape[0]
            if n_win > CENTRE_TARGET_K:
                lo = (n_win - CENTRE_TARGET_K) // 2
                sl = slice(lo, lo + CENTRE_TARGET_K)
                probs_t, logits_t, original_t = probs[sl], logits[sl], original_probs[sl]
"""

# inside the body, the reductions must read the (possibly sliced) arrays
BODY = [
    ("        if mode == 'max':\n            values[:, j] = probs[:, :, j].max(0).values\n",
     "        if mode == 'max':\n            values[:, j] = probs_t[:, :, j].max(0).values\n"),
    ("        elif mode == 'mean':\n            values[:, j] = probs[:, :, j].mean(0)\n",
     "        elif mode == 'mean':\n            values[:, j] = probs_t[:, :, j].mean(0)\n"),
    ("            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))\n",
     "            values[:, j] = torch.sigmoid(logits_t[:, :, j].mean(0))\n"),
    ("            values[:, j] = original_probs[:, :, j].mean(0)\n",
     "            values[:, j] = original_t[:, :, j].mean(0)\n"),
    ("            k = min(int(mode[3:]), probs.shape[0])\n            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)\n",
     "            k = min(int(mode[3:]), probs_t.shape[0])\n            values[:, j] = probs_t[:, :, j].topk(k, dim=0).values.mean(0)\n"),
]

# Only TTA_TARGET_POOL changes. PUBLIC_FRONTIER_TARGET_POOL is left exactly as the champion has it,
# so the public-frontier path is untouched and any board delta attributes to the main pooling path
# alone rather than to two things at once.
OLD_TTA = "TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}"
NEW_TTA = '''TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'centre_original_mean',
                   'ACL': 'centre_top2', 'MCL': 'centre_top2',
                   'Effusion': 'centre_mean', 'PF OA': 'centre_mean'}'''

nb = json.load(open(SRC, encoding="utf-8"))
counts = {}
edits = [("fn", OLD_FN, NEW_FN), ("tta", OLD_TTA, NEW_TTA)] + \
        [(f"body{i}", a, b) for i, (a, b) in enumerate(BODY)]
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for name, old, new in edits:
        if old in j:
            j = j.replace(old, new, 1)
            counts[name] = counts.get(name, 0) + 1
    c["source"] = j.splitlines(keepends=True)
missing = [n for n, _, _ in edits if counts.get(n, 0) != 1]
assert not missing, f"anchors not matched exactly once: {missing} (got {counts})"

for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse; anchors:", counts)

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)

old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0))
print(f"diff: +{sum(1 for l in d if l.startswith('+') and not l.startswith('+++'))} "
      f"-{sum(1 for l in d if l.startswith('-') and not l.startswith('---'))} lines")
print("built", OUT_DIR)
