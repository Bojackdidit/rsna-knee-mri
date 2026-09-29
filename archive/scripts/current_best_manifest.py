#!/usr/bin/env python3
"""Live manifest of our CURRENT-BEST RSNA assets, straight from the Kaggle API.

For each asset: exact ref, the files it actually contains, and its REAL visibility - taken from an
unauthenticated HTTP fetch, not from the API's privacy field. That field cannot be trusted here:
`kernels_list(mine=True)` returns is_private=False for notebooks that 404 to a logged-out request.

Prints a table and writes current_best_manifest.csv.
"""
import csv
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# The blended-label chain plus v11. These are what "current best" means: every arm behind the
# 0.938 board result and the unsubmitted v11 arm, plus the corpora and notebooks that build them.
DATASETS = [
    ("raptor-knee-fine96-blend-w", "v11 arm - BEST arm, gold OOF 0.9176 (not yet submitted)"),
    ("raptor-knee-fine96-blend", "96-slice corpus for v11"),
    ("raptor-knee-finespacing-blend", "v9blend arm - behind the 0.938 board best"),
    ("raptor-knee-v9blend-swa", "v9blend + SWA (scored 0.937)"),
    ("raptor-knee-labelblend", "v5blend arm / blended-label training set"),
    ("raptor-knee-labelblend4", "four-source blended labels"),
    ("raptor-knee-convnext-blend", "ConvNeXt-Small blended-label arm"),
    ("raptor-knee-swin-blend", "Swin-Base 384 blended-label arm"),
    ("raptor-knee-maxvit-blend", "MaxViT-Base 384 blended-label arm"),
    ("raptor-knee-widespan", "CoAtNet wide-span weights"),
]

KERNELS = [
    ("knee-mri-fine-spacing-blend-cross-team", "*** THE 0.938 SUBMISSION ***"),
    ("knee-mri-fine96-blend-cross-team", "v11 blend - built locally, may not exist on Kaggle yet"),
    ("knee-mri-fine-spacing-blend-swa-cross-team", "0.937"),
    ("knee-mri-blended-labels-cross-team-weight-60", "0.937 (blend weight 0.60)"),
    ("knee-mri-blended-labels-cross-team-weight-40", "blend weight 0.40"),
    ("knee-mri-cross-team-blend-with-blended-label-arm", "v5blend at 0.50"),
    ("knee-mri-four-source-blend-cross-team", "four-source labels"),
    ("knee-mri-two-arm-own-and-blended-labels", "two-arm pairing"),
    ("knee-mri-two-arm-fine-spacing-pair", "two-arm pairing"),
    ("knee-mri-three-architecture-cross-team", "three-family ensemble"),
    ("knee-mri-three-architecture-coatnet-heavy", "three-family, CoAtNet weighted"),
    ("knee-mri-coatnet-plus-convnext-cross-team", "two-family ensemble"),
    ("knee-mri-fine-spacing-own-labels-cross-team", "v9 own-labels cross-team"),
    ("knee-mri-blended-label-training", "blended-label TRAINING recipe"),
    ("knee-cross-team-blend-w65", "blend weight sweep"),
    ("knee-cross-team-blend-v5", "v5 cross-team"),
    ("knee-mri-efficient-single-arm-k24", "efficiency lane (0.929)"),
    ("knee-mri-efficient-single-arm-k42", "efficiency lane (0.930)"),
    ("knee-mri-fine-96-corpus-shard-0", "v11 corpus builder 0/5"),
    ("knee-mri-fine-96-corpus-shard-1", "v11 corpus builder 1/5"),
    ("knee-mri-fine-96-corpus-shard-2", "v11 corpus builder 2/5"),
    ("knee-mri-fine-96-corpus-shard-3", "v11 corpus builder 3/5"),
    ("knee-mri-fine-96-corpus-shard-4", "v11 corpus builder 4/5"),
]


def http(url):
    r = subprocess.run(["curl", "-s", "-o", os.devnull, "-w", "%{http_code}", "--max-time", "25",
                        url], capture_output=True, text=True)
    return r.stdout.strip()


def visibility(kind, ref):
    code = http("https://www.kaggle.com/%s/dreaddevelopment/%s"
                % ("datasets" if kind == "dataset" else "code", ref))
    return {"200": "PUBLIC", "404": "private"}.get(code, "HTTP " + code)


def dataset_files(api, ref):
    try:
        r = api.dataset_list_files("dreaddevelopment/" + ref)
        names = [str(f.name if hasattr(f, "name") else f) for f in getattr(r, "files", [])]
        return names or ["(none listed)"]
    except Exception as e:
        return ["(%s)" % type(e).__name__]


def kernel_users(api, ref):
    """Notebooks that attach this dataset - ours vs anyone else's."""
    try:
        ks = api.kernels_list(dataset="dreaddevelopment/" + ref, page_size=50)
        refs = [str(getattr(k, "ref", "")) for k in ks]
        ext = [r for r in refs if r and not r.startswith("dreaddevelopment/")]
        return len(refs), ext
    except Exception:
        return -1, []


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("authenticated to Kaggle API as: %s\n" % os.environ.get("KAGGLE_USERNAME",
                                                                  "(from kaggle.json)"))
    rows = []

    print("=" * 104)
    print("DATASETS  (current best)")
    print("=" * 104)
    print("%-34s %-9s %-6s %s" % ("exact dataset name", "visible", "ext", "files / note"))
    print("-" * 104)
    for ref, note in DATASETS:
        vis = visibility("dataset", ref)
        n, ext = kernel_users(api, ref)
        files = dataset_files(api, ref) if vis != "HTTP 000" else ["(unchecked)"]
        print("%-34s %-9s %-6s %s" % (ref, vis, len(ext), ", ".join(files)))
        print("%-34s %-9s %-6s   %s" % ("", "", "", note))
        rows.append(dict(kind="dataset", name="dreaddevelopment/" + ref, visibility=vis,
                         external_users=len(ext), files="; ".join(files), note=note,
                         url="https://www.kaggle.com/datasets/dreaddevelopment/" + ref))

    print()
    print("=" * 104)
    print("NOTEBOOKS  (current best)")
    print("=" * 104)
    print("%-52s %-9s %s" % ("exact notebook name", "visible", "note"))
    print("-" * 104)
    for ref, note in KERNELS:
        vis = visibility("kernel", ref)
        print("%-52s %-9s %s" % (ref, vis, note))
        rows.append(dict(kind="notebook", name="dreaddevelopment/" + ref, visibility=vis,
                         external_users="", files="", note=note,
                         url="https://www.kaggle.com/code/dreaddevelopment/" + ref))

    out = os.path.join(HERE, "current_best_manifest.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "name", "visibility", "external_users",
                                           "files", "note", "url"])
        w.writeheader()
        w.writerows(rows)

    pub = [r for r in rows if r["visibility"] == "PUBLIC"]
    odd = [r for r in rows if r["visibility"].startswith("HTTP")]
    print()
    print("=" * 104)
    print("%d assets checked | PUBLIC: %d | private: %d | inconclusive: %d"
          % (len(rows), len(pub), len(rows) - len(pub) - len(odd), len(odd)))
    for r in pub:
        print("   PUBLIC  %s" % r["name"])
    for r in odd:
        print("   %s  %s" % (r["visibility"], r["name"]))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
