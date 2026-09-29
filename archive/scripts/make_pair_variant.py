"""
Build a kernel variant that pairs the shipped champion with one new architecture.

Three architecture arms finish within hours of each other and each one needs the same four edits to
the same notebook: the ARMS list, the dataset it mounts, the kernel id, and the title. Doing that by
hand three times is where a wrong id silently overwrites the live champion, so it is scripted.

Usage:
  python make_pair_variant.py --tag CNXTL_E16 --arch convnext_large.fb_in22k_ft_in1k_384 \
         --dataset dreaddevelopment/raptor-knee-cnxtl-arm \
         --kernel dreaddevelopment/knee-mri-coatnet-plus-convnext-large \
         --title "Knee MRI - CoAtNet Plus ConvNeXt Large"
"""
import argparse, json, os, re, shutil

SRC = "variants/push_knee_mri_mixed_arch_pair"
CHAMP_DS = "dreaddevelopment/raptor-knee-scratch-aug2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="the arm's training tag, e.g. CNXTL_E16")
    ap.add_argument("--arch", required=True, help="timm arch string, must match the checkpoint")
    ap.add_argument("--dataset", required=True, help="kaggle dataset ref holding the new weights")
    ap.add_argument("--kernel", required=True, help="kaggle kernel ref, MUST be new")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out = a.out or f"variants/push_pair_{a.tag.lower()}"
    slug = a.kernel.split("/")[-1]

    # Kaggle derives the published slug from the TITLE, not from the id in kernel-metadata.json. Push
    # a title that slugifies differently and the notebook lands at a URL the id never named, so the
    # submit that follows aims at a kernel that does not exist. Caught once on EfficientNetV2-M, where
    # "EfficientNetV2-M" became "efficientnetv2-m" against an id saying "effnetv2m".
    want = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", a.title.lower())).strip("-")
    if want != slug:
        raise SystemExit(f"title slug '{want}' != kernel id '{slug}' - the push would land at the "
                         f"title slug. Rename one of them so they agree.")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    nb = json.load(open(os.path.join(SRC, "knee-mri-coatnet-plus-convnext.ipynb"), encoding="utf-8"))
    old = ('    {"file": "raptor_ft_CONVNEXT_E24.pt",  "arch": "convnext_base.fb_in22k_ft_in1k_384",'
           '     "res": 384, "w": 0.5},')
    new = f'    {{"file": "raptor_ft_{a.tag}.pt", "arch": "{a.arch}", "res": 384, "w": 0.5}},'
    hits = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        if old in src:
            hits += 1
            c["source"] = src.replace(old, new).splitlines(keepends=True)
    if hits != 1:
        raise SystemExit(f"ARMS line matched {hits} times, expected 1 - the source notebook moved")
    json.dump(nb, open(os.path.join(out, f"{slug}.ipynb"), "w", encoding="utf-8"))

    meta = json.load(open(os.path.join(SRC, "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = a.kernel
    meta["title"] = a.title
    meta["code_file"] = f"{slug}.ipynb"
    meta["dataset_sources"] = [CHAMP_DS, a.dataset]
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)

    print(f"wrote {out}")
    print(f"  kernel  {meta['id']}")
    print(f"  arms    raptor_ft_SCRATCH_AUG2.pt + raptor_ft_{a.tag}.pt ({a.arch})")
    print(f"  mounts  {meta['dataset_sources']}")


if __name__ == "__main__":
    main()
