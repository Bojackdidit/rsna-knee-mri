#!/usr/bin/env python3
"""N-arm kernel at 384 with MIXED architectures - the diversity lever that can actually ship.

The kernel prepares each study's windows ONCE and asserts "single pass needs one resolution", so
every arm must be 384 and must use full slots. Architecture is NOT constrained: load_model reads
"arch" out of the checkpoint and calls build_backbone generically, and the trainer writes "arch"
and "res" into every checkpoint it saves. That is why a second model FAMILY can ride here when a
224 arm or a slot-restricted arm cannot.

Arms are weighted equally and rank-averaged, exactly as the shipped pair is.

Usage:
    python make_arms384.py <slug> <dataset[,dataset...]> <arm1.pt:arch> [arm2.pt:arch ...]
        [--k N] [--title "..."]
    arch may be omitted for the CoAtNet default.
"""
import ast, json, os, sys

COAT = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
argv, K, TITLE = list(sys.argv[1:]), 48, None
out = []
i = 0
while i < len(argv):
    if argv[i] == "--k":
        K = int(argv[i + 1]); i += 2
    elif argv[i] == "--title":
        TITLE = argv[i + 1]; i += 2
    else:
        out.append(argv[i]); i += 1
SLUG, DATASETS, SPECS = out[0], out[1].split(","), out[2:]
assert SPECS, "no arms given"
ARMS = []
for s in SPECS:
    f, _, a = s.partition(":")
    ARMS.append((f, a or COAT))
assert len({f for f, _ in ARMS}) == len(ARMS), "duplicate arm"
TITLE = TITLE or ("Knee MRI - Raptor %d Arms at 384" % len(ARMS))

SRC = "variants/push_twogpu2/knee-mri-dual-gpu-decode.ipynb"
OUT_DIR = "variants/push_" + SLUG.replace("-", "_")

OLD = ('    {"file": "raptor_ft_confdist_swa.pt",          '
       '"arch": "%s", "res": 384, "w": 0.5},\n'
       '    {"file": "raptor_ft_coatnet_w80_full_swa.pt",  '
       '"arch": "%s", "res": 384, "w": 0.5},' % (COAT, COAT))
W = round(1.0 / len(ARMS), 6)
NEW = "\n".join('    {"file": "%s", "arch": "%s", "res": 384, "w": %s},' % (f, a, W)
                for f, a in ARMS)

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == 1, "two-arm block matched %d times (expected 1)" % hits

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for f, _ in ARMS:
    assert body.count(f) == 1, "arm not wired exactly once: %s" % f
assert body.count('"res": 384') == len(ARMS), "an arm is not at 384"
assert body.count('"w": %s' % W) == len(ARMS), "weights are not equal"
assert '"res": 224' not in body and '"res": 336' not in body and '"res": 480' not in body, \
    "an off-resolution arm slipped in; single pass needs one resolution"
# every speed win from the 29-minute kernel must survive
for marker in ("_hdr_one", "_px_one", "_prep_study", "_PARALLEL_ARMS", "[1pass]"):
    assert marker in body, "lost optimisation: %s" % marker
assert "_A5_SAVED" not in body and "_rad" + "_main" not in body, "a cut stage came back"
assert "K_EVAL = %d" % K in body, "K_EVAL is not %d" % K
assert "single pass needs one resolution" in body, "the one-resolution guard was removed"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations",
                                               "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("variants/push_twogpu2/kernel-metadata.json", encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
for d in DATASETS:
    if d and d not in meta["dataset_sources"]:
        meta["dataset_sources"].append(d)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("built %s" % OUT_DIR)
for f, a in ARMS:
    print("  %-38s %s" % (f, a))
print("  %d arms, w=%s each, K_EVAL %d" % (len(ARMS), W, K))
print("  datasets: %s" % meta["dataset_sources"])
