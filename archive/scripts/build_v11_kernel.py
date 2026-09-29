#!/usr/bin/env python3
"""Build the submission kernel for the v11-geometry arms. Zero GPU.

Prepared BEFORE the arms finish so boarding them is one command, not a scramble.

Two things this gets right that the champion kernel would get wrong for these weights:

1. **K_EVAL must be 24, not 48.** The v11 arms are trained with `--k 24 --k_eval 24`. Serving them
   at the champion's K=48 is exactly the train/serve mismatch that was identified and fixed in the
   trainer; re-introducing it at serve time would be the same class of error as the window
   re-spacing that scored 0.921 and the pooling swap that scored 0.922.
   See memory/serve-side-edits-are-refuted.

2. **Full slots.** These arms train on all five series (no `--slots`), so the kernel must build the
   full 96-slice volume - which is what it already does. No slot restriction.

Everything else is left byte-identical to the champion so the board read attributes to the arms.

--arms takes any list of trained tags, and repeating one self-pairs it (measured +0.0040 free,
memory/best-single-arm-is-not-best-pair). Runtime is set by the BUSY CARD: the kernel puts arm i on
cuda:i%2, so 2 arms at K=24 and 4 arms at K=24 both cost 48 window passes on the busy card - the
champion's own cost, ~23.8 min. Six arms cost 72 passes, ~30 min.

Usage:
    python build_v11_kernel.py --arms A,A           # first read: arm A self-paired
    python build_v11_kernel.py --arms A,B,C,D       # the 4-fold ensemble
"""
import argparse, copy, json, os, re

BASE_NB = os.path.join("variants", "push_fastdcm_clean",
                       "knee-mri-raptor-direct-dicom-read.ipynb")
BASE_META = os.path.join("variants", "push_fastdcm_clean", "kernel-metadata.json")

OLD_ARMS_START = 'ARMS = [\n    {"file": "raptor_ft_confdist_swa.pt",'
OLD_ARMS_END = '"res": 384, "w": 0.5},\n]'
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"

OLD_K = "K_EVAL = 48   # every window position the volume holds, not an evenly spaced subset"
NEW_K = ("K_EVAL = 24   # MUST match the arms' k_eval. These were trained --k 24 --k_eval 24; "
         "serving them\n              # at 48 is the train/serve mismatch the trainer fixed. "
         "Do not 'improve' this.")


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
    ap.add_argument("--arms", default="A,B", help="comma list of V11_AUG tags; repeat one to self-pair")
    ap.add_argument("--files", default="", help="explicit .pt filenames, comma separated; overrides --arms")
    ap.add_argument("--dataset", default="dreaddevelopment/raptor-knee-v11-aug-arms")
    ap.add_argument("--slug", default="", help="override the kernel slug")
    ap.add_argument("--title", default="", help="override the kernel title; must slugify to --slug")
    ap.add_argument("--keval", type=int, default=24,
                    help="windows per study. The fastdcm base does NOT interleave - both arms read "
                         "the SAME window set - so this is TOTAL coverage, not a per-arm budget. "
                         "Use 48 to match the champion when the weights are the variable under test.")
    a = ap.parse_args()

    if a.files:
        files = [f.strip() for f in a.files.split(",") if f.strip()]
        tags = [os.path.basename(f)[len("raptor_ft_"):-3] if f.startswith("raptor_ft_") else
                os.path.basename(f)[:-3] for f in files]
    else:
        tags = [t.strip().upper() for t in a.arms.split(",") if t.strip()]
        if not tags:
            raise SystemExit("--arms is empty")
        files = ["raptor_ft_V11_AUG_%s.pt" % t for t in tags]
    w = round(1.0 / len(tags), 4)
    new_arms = "ARMS = [\n" + "".join(
        '    {"file": "%s", "arch": "%s", "res": 384, "w": %s},\n' % (f, ARCH, w) for f in files
    ) + "]"

    busy = (len(tags) + 1) // 2          # arms on the fuller of the two cards
    print("[plan] %d arm(s) %s -> busy card carries %d x K24 = %d window passes/study; "
          "the champion pays 48 for 23.8 min, so expect roughly %.0f%% of that model time"
          % (len(tags), "+".join(tags), busy, busy * 24, 100.0 * busy * 24 / 48))

    missing = sorted({f for f in files if not os.path.exists(os.path.join("arms_v11", f))})
    if missing:
        print("NOTE weights not pulled yet: %s" % ", ".join(missing))
        print("     building the kernel anyway; upload the dataset before pushing.")

    base = json.load(open(BASE_NB, encoding="utf-8"))
    base_bad = bad_cells(base, BASE_NB)
    nb = copy.deepcopy(base)

    whole = None
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        i = s.find(OLD_ARMS_START)
        if i >= 0:
            j = s.find(OLD_ARMS_END, i)
            assert j > 0, "could not find end of ARMS block"
            whole = s[i:j + len(OLD_ARMS_END)]
            break
    assert whole, "ARMS block not found"
    patch(nb, whole, new_arms, "ARMS")
    if a.keval != 48:
        patch(nb, OLD_K, NEW_K.replace("K_EVAL = 24", "K_EVAL = %d" % a.keval), "K_EVAL")
    else:
        print("[plan] K_EVAL left at 48 - the champion's coverage, so the weights are the only variable")

    new_bad = sorted(set(bad_cells(nb, "v11")) - set(base_bad))
    if new_bad:
        raise SystemExit("patch introduced syntax errors in cells %s" % new_bad)

    full = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for f in files:
        assert f in full, "arm %s did not make it into the notebook" % f
    assert "K_EVAL = %d" % a.keval in full, "K_EVAL not set to %d" % a.keval
    # This guard catches a patch that silently failed and left the champion's ARMS block in place.
    # It must not fire when a champion arm is deliberately part of the requested line-up.
    if "raptor_ft_confdist_swa.pt" not in files:
        assert "raptor_ft_confdist_swa.pt" not in full, "champion arm survived the patch"

    # The slug must be UNIQUE per arm set. memory/kernel-metadata-inherits-the-source-id: a variant
    # that reuses an id silently overwrites whatever lives there - once that was the live champion.
    suffix = "-".join(re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") for t in tags)
    slug = a.slug or ("knee-mri-coatnet-384-v11-" + suffix)
    d = os.path.join(a.out, "push_" + slug.replace("-", "_"))
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, slug + ".ipynb"), "w", encoding="utf-8"))
    m = dict(json.load(open(BASE_META, encoding="utf-8")))
    m.pop("id_no", None)
    m["id"] = "dreaddevelopment/" + slug
    m["title"] = a.title or ("Knee MRI - CoAtNet 384 V11 " + suffix.upper().replace("-", " "))
    m["code_file"] = slug + ".ipynb"
    m["kernel_type"] = "notebook"
    m["dataset_sources"] = [a.dataset]
    json.dump(m, open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("[ok] %s -> %s" % (slug, d))
    print("     arms: %s | K_EVAL %d | full slots | dataset %s" % (", ".join(files), a.keval, a.dataset))
    # REAL check: Kaggle files the kernel under the slug of the TITLE. If that differs from the id
    # in kernel-metadata.json the push succeeds but lands at a different ref, and the later
    # `competitions submit -k <id>` targets a kernel that does not exist. Hit exactly this on the
    # PF OA probe.
    tslug = re.sub(r"[^a-z0-9]+", "-", m["title"].lower()).strip("-")
    if tslug != slug:
        raise SystemExit("TITLE/ID MISMATCH: title %r slugs to %r but id is %r"
                         % (m["title"], tslug, slug))
    print("     title slug == id: %s  (verified, not assumed)" % tslug)


if __name__ == "__main__":
    main()
