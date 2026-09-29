#!/usr/bin/env python3
"""fastdcm reads + arm-only CoAtNet + the LEGACY second family as the outside half.

This configuration has never existed. The champion blend runs a 20-member DINOv2 partner half that
[[rsna-partner-half-is-dead-weight]] showed is worthless (arm-only 0.941 vs 0.940 at 1/6 the
compute), and the arm-only kernel drops the outside half entirely - seeding submission.csv from
sample_submission, i.e. a constant 0.5. So the V18 blend currently mixes our CoAtNets with NOTHING.

The legacy family is the piece worth having there: its per-target weights are Lateral Meniscus 15.0,
Lateral OA 15.0, Contusion 5.0, Medial OA 2.5 - almost exactly our weakest findings - and it decodes
at its own geometry so it disagrees with our stack by construction. It was stripped for RUNTIME, and
fastdcm cut the decode cost ~2.4x, which is what makes this buildable now.

Three edits: restore the loader, make the partner manifest OPTIONAL so `groups` ends up holding only
the legacy geometry, and call infer_from_package where the arm-only kernel used to stub the half out.
"""
import io, json, os, sys

SRC = "variants/push_fastdcm/knee-mri-raptor-direct-dicom-read.ipynb"
OUT_DIR = "variants/push_legacy_arm"
SLUG = "knee-mri-second-family-arm-only"
TITLE = "Knee MRI - Second Model Family With The Arm Alone"

LOADER = io.open("_legacy_patch.txt", encoding="utf-8").read().rstrip("\n")
nb = json.load(io.open(SRC, encoding="utf-8"))

STUB = "def legacy_group_members():\n    return {}\n"
OLD_MAN = ("    man = json.loads((Path(path) / 'manifest.json').read_text())\n"
           "    members = man['members']\n")
NEW_MAN = ("    # Partner package is OPTIONAL now. With it absent, `groups` below ends up holding only\n"
           "    # the legacy geometry, so this runs the second family ALONE - which is the point:\n"
           "    # the 20-member partner half is dead weight (arm-only 0.941 vs 0.940 at 1/6 compute).\n"
           "    _manp = Path(path) / 'manifest.json'\n"
           "    members = json.loads(_manp.read_text())['members'] if _manp.is_file() else []\n")
OLD_SEED = ("print('ONEPASS: partner half not run', flush=True)\n"
            "import pandas as _ao_pd\n"
            "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')\n"
            "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)\n")
NEW_SEED = ("print('ONEPASS: 20-member partner half NOT run; LEGACY second family IS run', flush=True)\n"
            "import pandas as _ao_pd\n"
            "_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')\n"
            "_ao_df.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "try:\n"
            "    # no manifest at this path -> members=[] -> only the legacy decode group runs.\n"
            "    # bank() writes submission.csv as members land, which V18 then blends with the arms.\n"
            "    infer_from_package(ASSET / 'rsna-knee-weights', DEVS[0])\n"
            "    print('LEGACY: second family completed', flush=True)\n"
            "except Exception as _e:\n"
            "    # never let the outside half take the run down - the arm alone already scores 0.941\n"
            "    print(f'LEGACY: skipped ({type(_e).__name__}: {_e})', flush=True)\n"
            "    _ao_df.to_csv('/kaggle/working/submission.csv', index=False)\n")

hits = {"stub": 0, "man": 0, "seed": 0}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    orig = s
    if STUB in s:
        s = s.replace(STUB, "def legacy_group_members():\n" + LOADER + "\n", 1); hits["stub"] += 1
    if OLD_MAN in s:
        s = s.replace(OLD_MAN, NEW_MAN, 1); hits["man"] += 1
    if OLD_SEED in s:
        s = s.replace(OLD_SEED, NEW_SEED, 1); hits["seed"] += 1
    if s != orig:
        cell["source"] = s.splitlines(keepends=True)

print("patch hits:", hits)
if hits["stub"] != 1 or hits["man"] != 1 or hits["seed"] != 1:
    sys.exit("FAILED: expected exactly one hit each; anchors did not match")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"), ensure_ascii=False)
meta = json.load(io.open("variants/push_fastdcm/kernel-metadata.json", encoding="utf-8"))
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
meta.pop("id_no", None)
ds = list(meta.get("dataset_sources", []))
if "tonylica/rsna2026-models" not in ds:
    ds.append("tonylica/rsna2026-models")
meta["dataset_sources"] = ds
json.dump(meta, io.open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)

chk = "".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
for name, cond in [
    ("loader restored",        "LEGACY_MEMBER_WEIGHT_BY_TARGET" in chk and "def find_legacy_bundles" in chk),
    ("stub gone",              "def legacy_group_members():\n    return {}" not in chk),
    ("manifest optional",      "_manp.is_file() else []" in chk),
    ("legacy actually called", "infer_from_package(ASSET / 'rsna-knee-weights', DEVS[0])" in chk),
    ("guarded by try/except",  "LEGACY: skipped" in chk),
    ("fastdcm still present",  "def parse_header(buf)" in chk and "m, a = _fd_read_one(f)" in chk),
    ("arms untouched",         "raptor_ft_confdist_swa.pt" in chk and "raptor_ft_coatnet_w80_full_swa.pt" in chk),
    ("bundles dataset added",  "tonylica/rsna2026-models" in json.dumps(meta)),
]:
    print("  %-24s %s" % (name, "OK" if cond else "FAIL"))
import ast
for c in nb["cells"]:
    if c["cell_type"] == "code":
        try: ast.parse("".join(c["source"]))
        except SyntaxError as e: print("  SYNTAX ERROR in a cell:", e); break
else:
    print("  all cells parse         OK")
print("built:", os.path.join(OUT_DIR, SLUG + ".ipynb"))
