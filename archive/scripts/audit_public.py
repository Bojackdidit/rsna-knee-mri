#!/usr/bin/env python3
"""SAFETY AUDIT: what of ours is PUBLIC, and does any of it carry our current best?

Standing order: no public weight release until we are top 10-20, and then only a ~0.927-class set.
The reasoning is competitive - our public 0.924 already pulled 103 teams up to ~0.935, so releasing
our current best would lift the field past us.

This enumerates EVERY dataset, notebook and model on the account with its real privacy flag from
the API (download and vote counts are NOT a privacy signal - a private asset can show votes from
before it was made private, and a public one can show zero). It then flags anything public whose
name or file list suggests it carries model weights.

Read-only. Changes nothing. Prints a verdict per asset so each can be checked by eye.
"""
import json
import os
import re
import sys

KNEE = re.compile(r"knee|raptor|rsna|coatnet|dinov|twelve|maxvit|swin|convnext|cnv2|effv2|cnn336",
                  re.I)
WEIGHTY = re.compile(r"\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.bin$|\.h5$|\.onnx$", re.I)


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def truthy_private(obj):
    """Kaggle's client has used isPrivate / is_private / privacy across versions."""
    for attr in ("isPrivate", "is_private", "privacy"):
        v = getattr(obj, attr, None)
        if v is None:
            continue
        if isinstance(v, str):
            return v.strip().lower() in ("private", "true")
        return bool(v)
    return None


def main():
    a = api()
    problems, checked = [], 0

    print("=" * 100)
    print("DATASETS")
    print("=" * 100)
    seen = set()
    for page in range(1, 8):
        try:
            rows = a.dataset_list(mine=True, page=page)
        except TypeError:
            rows = a.dataset_list(user=os.environ.get("KAGGLE_USERNAME", ""), page=page)
        if not rows:
            break
        for d in rows:
            ref = str(d.ref)
            if ref in seen:
                continue
            seen.add(ref)
            priv = truthy_private(d)
            checked += 1
            knee = bool(KNEE.search(ref) or KNEE.search(str(getattr(d, "title", ""))))
            tag = "PRIVATE" if priv else ("PUBLIC " if priv is False else "UNKNOWN")
            mark = ""
            if priv is False and knee:
                mark = "   <-- PUBLIC and knee-related"
                problems.append(("dataset", ref, str(getattr(d, "title", ""))))
            if knee or priv is False:
                print("%-8s %-52s %s%s" % (tag, ref, getattr(d, "title", ""), mark))

    print()
    print("=" * 100)
    print("NOTEBOOKS / KERNELS")
    print("=" * 100)
    seen = set()
    for page in range(1, 8):
        rows = a.kernels_list(mine=True, page=page, page_size=100)
        if not rows:
            break
        for k in rows:
            ref = str(getattr(k, "ref", ""))
            if not ref or ref in seen:
                continue
            seen.add(ref)
            checked += 1
            knee = bool(KNEE.search(ref) or KNEE.search(str(getattr(k, "title", ""))))
            priv = truthy_private(k)
            tag = "PRIVATE" if priv else ("PUBLIC " if priv is False else "UNKNOWN")
            mark = ""
            if priv is False and knee:
                mark = "   <-- PUBLIC and knee-related"
                problems.append(("kernel", ref, str(getattr(k, "title", ""))))
            if knee or priv is False:
                print("%-8s %-52s %s%s" % (tag, ref, getattr(k, "title", ""), mark))

    print()
    print("=" * 100)
    print("SUMMARY  (%d assets enumerated)" % checked)
    print("=" * 100)
    if not problems:
        print("No PUBLIC knee/RSNA asset found by the API listing.")
    else:
        print("%d PUBLIC knee/RSNA assets - each needs its CONTENTS checked:" % len(problems))
        for kind, ref, title in problems:
            print("   %-8s %-52s %s" % (kind, ref, title))
    json.dump([{"kind": k, "ref": r, "title": t} for k, r, t in problems],
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "public_knee_assets.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
