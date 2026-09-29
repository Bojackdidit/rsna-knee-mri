#!/usr/bin/env python3
"""Build a NEW kernel variant from the live champion, without overwriting it.

On 2026-09-22 a variant was built by pulling the champion's metadata, changing `id` and `title`,
and diffing the notebook to prove only two lines moved. It still overwrote the live champion,
because `kernel-metadata.json` also carries **`id_no`** and Kaggle resolves by that number, not by
`id`. The push renamed the champion and replaced its code.

So this script does three things by construction:
  - DELETES `id_no`, which is the actual bug;
  - refuses a title whose slug does not equal the new id (Kaggle derives the published URL from the
    title, so a mismatch lands the notebook where the submit cannot find it);
  - prints the diff against the source so the change is always seen before it ships.

`--edit` is a LITERAL substitution applied to every match, so an anchor like `"w": 0.5},` hits both
arms. Use a full distinctive line as the anchor, and read the printed diff - that is the net.

Usage:
  python make_variant_safe.py --src <dir with pulled champion> --out variants/push_x \
      --id dreaddevelopment/knee-mri-x --title "Knee MRI X" --edit OLD=NEW [--edit ...]
"""
import argparse
import difflib
import glob
import json
import os
import re
import shutil


def code_of(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--edit", action="append", default=[], help="OLD=NEW literal substitution")
    a = ap.parse_args()

    slug = a.id.split("/")[-1]
    want = re.sub(r"[^a-z0-9]+", "-", a.title.lower()).strip("-")
    if want != slug:
        raise SystemExit("[refuse] title %r slugifies to %r but id slug is %r - they must match"
                         % (a.title, want, slug))

    src_nb = glob.glob(os.path.join(a.src, "*.ipynb"))[0]
    nb = json.load(open(src_nb, encoding="utf-8"))
    before = code_of(nb)

    n = 0
    for e in a.edit:
        old, new = e.split("=", 1)
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            if old in s:
                c["source"] = s.replace(old, new).splitlines(keepends=True)
                n += 1
    if a.edit and n == 0:
        raise SystemExit("[refuse] no edit matched - the anchor text is wrong, nothing would change")

    os.makedirs(a.out, exist_ok=True)
    out_nb = os.path.join(a.out, slug + ".ipynb")
    json.dump(nb, open(out_nb, "w", encoding="utf-8"))

    m = json.load(open(os.path.join(a.src, "kernel-metadata.json"), encoding="utf-8"))
    dropped = m.pop("id_no", None)
    m["id"] = a.id
    m["title"] = a.title
    m["code_file"] = os.path.basename(out_nb)
    json.dump(m, open(os.path.join(a.out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)

    print("[variant] %s" % a.out)
    print("  id      : %s" % m["id"])
    print("  title   : %s  (slug %s - matches)" % (m["title"], want))
    print("  id_no   : DROPPED (was %s) <- this is what prevents overwriting the source" % dropped)
    print("  datasets: %s" % m.get("dataset_sources"))
    d = [l for l in difflib.unified_diff(before.split("\n"), code_of(nb).split("\n"), lineterm="", n=0)
         if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    print("  code diff (%d lines):" % len(d))
    for l in d:
        print("    " + l[:150])
    print("\nAfter pushing, ALWAYS verify the source kernel still resolves:")
    print("  kaggle kernels status %s" % m.get("id", "").rsplit("/", 1)[0] + "/<original-slug>")


if __name__ == "__main__":
    main()
