#!/usr/bin/env python3
"""Definitive visibility check: fetch every knee/RSNA asset UNAUTHENTICATED.

WHY NOT THE API
`kernels_list(mine=True)` reports `is_private = False` for kernels that are demonstrably private -
`knee-mri-fine-spacing-own-labels-cross-team` comes back False from the API but 404s to a logged-out
request. Trusting it would have reported our entire blend recipe as leaked when it is not, and
worse, the same unreliability could hide a real leak in the other direction.

The only answer that matters is what a competitor sees, so that is what this measures: a plain HTTP
GET with no credentials. 200 = they can read it. 404 = they cannot.

Read-only. Sends nothing but the URL.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KNEE = re.compile(r"knee|raptor|rsna|coatnet|dinov|twelve|maxvit|swin|convnext|cnv2|effv2|cnn336",
                  re.I)


def status(url):
    r = subprocess.run(["curl", "-s", "-o", os.devnull, "-w", "%{http_code}",
                        "--max-time", "25", url], capture_output=True, text=True)
    return r.stdout.strip()


def collect():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    ds, ks = [], []
    for page in range(1, 8):
        rows = a.dataset_list(mine=True, page=page)
        if not rows:
            break
        ds += [str(d.ref) for d in rows]
    for page in range(1, 8):
        rows = a.kernels_list(mine=True, page=page, page_size=100)
        if not rows:
            break
        ks += [str(k.ref) for k in rows if getattr(k, "ref", None)]
    return sorted(set(ds)), sorted(set(ks))


def main():
    ds, ks = collect()
    ds = [r for r in ds if KNEE.search(r)]
    ks = [r for r in ks if KNEE.search(r)]
    print("knee/RSNA assets to verify: %d datasets, %d notebooks\n" % (len(ds), len(ks)))

    public, private, other = [], [], []
    for kind, refs, base in (("dataset", ds, "datasets"), ("notebook", ks, "code")):
        print("=" * 92)
        print(kind.upper())
        print("=" * 92)
        for ref in refs:
            code = status("https://www.kaggle.com/%s/%s" % (base, ref))
            label = {"200": "PUBLIC", "404": "private"}.get(code, "HTTP " + code)
            (public if code == "200" else private if code == "404" else other).append((kind, ref))
            print("  %-8s %s" % (label, ref))
        print()

    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    print("visible to competitors : %d" % len(public))
    for kind, ref in public:
        print("    %-9s %s" % (kind, ref))
    print("not visible            : %d" % len(private))
    if other:
        print("INCONCLUSIVE (re-check) : %d" % len(other))
        for kind, ref in other:
            print("    %-9s %s" % (kind, ref))
    json.dump({"public": [r for _, r in public], "private": [r for _, r in private],
               "inconclusive": [r for _, r in other]},
              open(os.path.join(HERE, "visibility.json"), "w"), indent=1)
    print("\nwrote visibility.json")


if __name__ == "__main__":
    sys.exit(main())
