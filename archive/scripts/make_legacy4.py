#!/usr/bin/env python3
"""Restore the legacy member group that our reproduction stripped out.

The public cross-team notebooks load a SECOND model family through `legacy_group_members()`:
four DINOv2-small fold models from tonylica/rsna2026-models, fitted at a genuinely different
geometry - 224 px, 160 mm crop, band 0.2-0.8, 9 slices (7 window positions), legacy pixel rules -
so they decode as their own group and disagree with the main stack by construction.

Our champion has that function stubbed to `return {}` and does not even attach the dataset. Every
supporting piece is still in our file (RULES_LEGACY, legacy_fold_soft_window_pool,
LEGACY_FOLD_SOFTPOOL_ALPHA, per-target weight handling in the combiner) - only the loader was
removed, almost certainly the "runtime_members_removed: 5" note in the notebook metadata. Runtime
is no longer a reason: our measured run is 3 h 32 m against an 8 h budget.

Why this is the right thing to restore rather than more TTA: the legacy members carry weight on
only four findings - Lateral Meniscus 15.0, Lateral OA 15.0, Contusion 5.0, Medial OA 2.5 - and
those are almost exactly the findings our own per-finding OOF says we are weakest at (Lateral OA
0.85, Lateral Meniscus 0.89, Contusion 0.87). It adds real trained models where we actually lose.

Verified before building: the bundle's slot contract is the RECOVERED 6-slot scheme, which is the
scheme our kernel runs by default, so the loader's compatibility guard passes and this genuinely
activates instead of silently skipping.
"""
import ast, difflib, json, os, re

CHAMP = "champ/knee-mri-blend-even-full.ipynb"
PUB = "survey2/mattiaangeli_bend-the-knee-to-dinov3-ensembled/_src.py"
OUT_DIR = "variants/push_legacy4"
SLUG = "knee-mri-second-family-four-runs"
TITLE = "Knee MRI - Second Model Family, Four Runs"

pub = open(PUB, encoding="utf-8").read()


def block(start, end):
    i = pub.index(start)
    j = pub.index(end, i)
    return pub[i:j]


weights_const = block("LEGACY_MEMBER_WEIGHT_BY_TARGET = {", "\n\ndef window_starts").strip() + "\n"
loader = block("LEGACY_BUNDLE_FILE = 'rsna_20260807_v1.pt'", "\ndef _run_member(")
OLD_FIND = "def find_legacy_bundle():\n    base = Path('/kaggle/input')\n    if not base.is_dir():\n        return None\n    for root, dirs, files in os.walk(base):\n        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]\n        if LEGACY_BUNDLE_FILE in files:\n            return Path(root) / LEGACY_BUNDLE_FILE\n    return None"
NEW_FIND = "def find_legacy_bundles():\n    # All four published runs, not just v1. They share one geometry, so they cost a single\n    # extra decode group; only the member forwards are added.\n    base = Path('/kaggle/input')\n    found = []\n    if not base.is_dir():\n        return found\n    for root, dirs, files in os.walk(base):\n        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]\n        for fn in sorted(files):\n            if re.fullmatch(r'rsna_20260807_v\\d+\\.pt', fn):\n                found.append(Path(root) / fn)\n    return sorted(found)"

loader = loader.replace(OLD_FIND, NEW_FIND)
loader = loader.replace("""def legacy_group_members():
    p = find_legacy_bundle()
    if p is None:
        log('no legacy bundle attached; blending skipped')
        return {}
    try:
        b = torch.load(p, map_location='cpu', weights_only=False)""",
"""def legacy_group_members():
    paths = find_legacy_bundles()
    if not paths:
        log('no legacy bundle attached; blending skipped')
        return {}
    out_ms, key = ([], None)
    for p in paths:
      try:
        b = torch.load(p, map_location='cpu', weights_only=False)""")
loader = loader.replace("""        b_slots = [tuple(s)[0] for s in b.get('slots', SLOTS)]
        if list(b.get('targets', TARGETS)) != TARGETS or b_slots != [s[0] for s in SLOTS]:
            log(f'legacy bundle {p.name}: target/slot contract differs; blending skipped')
            return {}""",
"""        b_slots = [tuple(s)[0] for s in b.get('slots', SLOTS)]
        if list(b.get('targets', TARGETS)) != TARGETS or b_slots != [s[0] for s in SLOTS]:
            log(f'legacy bundle {p.name}: target/slot contract differs; skipped')
            continue""")
loader = loader.replace("'id': f\"legacy-f{f.get('fold', k)}\"",
                        "'id': f\"legacy-{p.stem.split('_')[-1]}-f{f.get('fold', k)}\"")
loader = loader.replace("LEGACY_MEMBER_WEIGHT_BY_TARGET.get(t, 0.0)",
                        "LEGACY_MEMBER_WEIGHT_BY_TARGET.get(t, 0.0) / max(len(paths), 1)")
loader = loader.replace("""        if ms:
            active = sorted(set(LEGACY_MEMBER_WEIGHT_BY_TARGET.values()))
            log(f'legacy bundle {p.name}: {len(ms)} fold(s) join with target-specific per-member weights {active}')
        return {key: ms} if ms else {}
    except Exception as exc:
        log(f'legacy bundle unusable ({type(exc).__name__}: {exc}); blending skipped')
        return {}""",
"""        out_ms.extend(ms)
        log(f'legacy bundle {p.name}: {len(ms)} fold(s) joined')
      except Exception as exc:
        log(f'legacy bundle {p.name} unusable ({type(exc).__name__}: {exc}); skipped')
    if out_ms:
        log(f'legacy family: {len(out_ms)} member(s) from {len(paths)} run(s); '
            f'per-member target weights divided by {len(paths)} so the family share is unchanged')
    return {key: out_ms} if out_ms and key else {}""")
assert 'find_legacy_bundles' in loader and 'out_ms' in loader, "legacy4 rewrite did not apply"

print("lifted verbatim from the public kernel:")
print(f"  per-target weights : {weights_const.strip()[:110]}")
print(f"  loader block       : {len(loader)} chars, "
      f"{loader.count('def ')} functions")

EDITS = [
    # per-target weights, placed next to the softpool alpha table it belongs with
    (None, "LEGACY_FOLD_SOFTPOOL_ALPHA", weights_const),
    # the loader + real implementation, replacing the stub
    ("def legacy_group_members():\n    return {}\n", None, loader),
]

nb = json.load(open(CHAMP, encoding="utf-8"))
done = [0, 0]
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    # 1. insert the weights constant after the softpool alpha dict block
    if not done[0] and "LEGACY_FOLD_SOFTPOOL_ALPHA" in j:
        m = re.search(r"LEGACY_FOLD_SOFTPOOL_ALPHA = \{[^{}]*\}\n", j, re.S)
        assert m, "could not find the LEGACY_FOLD_SOFTPOOL_ALPHA block to anchor to"
        j = j[:m.end()] + weights_const + j[m.end():]
        done[0] = 1
    # 2. replace the stub with the real loader
    stub = "def legacy_group_members():\n    return {}\n"
    if not done[1] and stub in j:
        j = j.replace(stub, loader, 1)
        done[1] = 1
    c["source"] = j.splitlines(keepends=True)
assert done == [1, 1], f"edits applied: {done}"

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
assert "tonylica/rsna2026-models" not in meta["dataset_sources"]
meta["dataset_sources"].append("tonylica/rsna2026-models")
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("attached dataset:", meta["dataset_sources"][-1])

old = "".join("".join(c["source"]) for c in json.load(open(CHAMP, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0))
print(f"diff: +{sum(1 for l in d if l.startswith('+') and not l.startswith('+++'))} "
      f"-{sum(1 for l in d if l.startswith('-') and not l.startswith('---'))} lines")
