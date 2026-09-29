#!/usr/bin/env python3
"""Assemble the v11 96-slice corpus on the box from the five Kaggle shard kernels.

WHY THIS EXISTS
---------------
2026-09-13: train and serve were running different geometries and had been for weeks.
    _setup_box.sh:28    assert v.shape == (4407, 44, 336, 336)   <- what we TRAINED on
    every serve kernel  SLOTS 27/21/18/12/18, MAXS = 96          <- what we SERVE
A window is three physically ADJACENT CORPUS slices stacked as RGB, so at 44 slices that
triplet spans 2.18x MORE anatomy than the triplet the kernel hands the model at test time.
Across twelve board reads and eight architectures, no box-trained arm ever cleared 0.928
while the two v11-matched arms give 0.941.

The v11 corpus does NOT need rebuilding. It was built on 2026-08-26 and is complete:
5 shards x 5 chunks, 4,407 studies, 0 failures, span 0.02-0.98. This script pulls those
kernel outputs straight from the Kaggle CDN to the box and concatenates them - no home
bandwidth, no 20 GB /kaggle/working cap, no new dataset to create.

Peak disk is ~50 GB, not ~96, because each chunk is deleted as soon as it is copied in.
The output is written through a memmap so the box never needs 47.8 GB of RAM.

Usage (on the box):
    python _merge_v11.py --out /workspace/knee/corpus_v11
    python _merge_v11.py --out /workspace/knee/corpus_v11 --keep-chunks   # debugging only
"""
import argparse, json, os, subprocess, time
import numpy as np

NSHARD = 5
MAXS = 96
IMG = 336
N_STUDIES = 4407
SLOT_BUDGETS = [27, 21, 18, 12, 18]          # Sag-FS, Sag-nonFS, Cor-FS, Cor-nonFS, Axial
KERNEL = "dreaddevelopment/knee-mri-fine-96-corpus-shard-%d"


def sh(cmd):
    print("  $ %s" % cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit("command failed (%d): %s" % (r.returncode, cmd))


def pull(shard, dest, pattern):
    sh('kaggle kernels output %s -p "%s" --file-pattern "%s" -q' % (KERNEL % shard, dest, pattern))


def fetch_shard_meta(shard, stage):
    """Pull one shard's small files. Returns (meta_dict, dir, n_chunks)."""
    d = os.path.join(stage, "shard%d" % shard)
    os.makedirs(d, exist_ok=True)
    meta_p = os.path.join(d, "v11_chunks_%d.json" % shard)
    if not os.path.exists(meta_p):
        pull(shard, d, "v11_(chunks|meta)_%d\\.json" % shard)
    meta = json.load(open(meta_p))
    for name in ("v11_masks_%d.npy" % shard, "v11_ids_%d.npy" % shard):
        if not os.path.exists(os.path.join(d, name)):
            pull(shard, d, name.replace(".", "\\."))
    return meta, d, int(meta["n_chunks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="destination dir for all_vols/all_masks/all_ids")
    ap.add_argument("--stage", default="", help="scratch dir for chunks (default <out>/_stage)")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="do NOT delete each chunk after copying (needs ~96 GB instead of ~50)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    stage = a.stage or os.path.join(a.out, "_stage")
    os.makedirs(stage, exist_ok=True)

    # ---- plan first: read every shard's metadata before moving a single byte ----
    plan = []
    total = 0
    for s in range(NSHARD):
        meta, d, n_chunks = fetch_shard_meta(s, stage)
        rows = int(meta["rows"])
        shape = [int(x) for x in meta["shape"]]
        if shape[1:] != [MAXS, IMG, IMG]:
            raise SystemExit("shard %d is %r - not the 96-slice v11 geometry" % (s, shape))
        plan.append((s, d, n_chunks, int(meta["chunk_rows"]), rows))
        total += rows
        print("[plan] shard %d: %d rows, %d chunks of %d, shape %r"
              % (s, rows, n_chunks, meta["chunk_rows"], shape), flush=True)
    if total != N_STUDIES:
        raise SystemExit("shards total %d rows, expected %d" % (total, N_STUDIES))
    print("[plan] %d studies, %.2f GB uncompressed"
          % (total, total * MAXS * IMG * IMG / 1e9), flush=True)

    # ---- allocate the output through a memmap: never holds 47.8 GB in RAM ----
    vols_p = os.path.join(a.out, "all_vols.npy")
    vols = np.lib.format.open_memmap(vols_p, mode="w+", dtype=np.uint8,
                                     shape=(total, MAXS, IMG, IMG))
    masks = np.zeros((total, MAXS), np.uint8)
    ids = []
    row = 0
    for s, d, n_chunks, chunk_rows, rows in plan:
        m = np.load(os.path.join(d, "v11_masks_%d.npy" % s))
        i = np.load(os.path.join(d, "v11_ids_%d.npy" % s), allow_pickle=True).astype(str)
        if len(m) != rows or len(i) != rows:
            raise SystemExit("shard %d: masks/ids are %d/%d, expected %d"
                             % (s, len(m), len(i), rows))
        masks[row:row + rows] = m
        ids.extend(i.tolist())
        got = 0
        for c in range(n_chunks):
            name = "v11_vols_%d_c%d.npy" % (s, c)
            p = os.path.join(d, name)
            if not os.path.exists(p):
                pull(s, d, name.replace(".", "\\."))
            part = np.load(p, mmap_mode="r")
            if part.shape[1:] != (MAXS, IMG, IMG):
                raise SystemExit("%s is %r" % (name, part.shape))
            vols[row + got: row + got + len(part)] = part
            got += len(part)
            del part
            if not a.keep_chunks:
                os.remove(p)
            print("  shard %d chunk %d/%d -> %d rows placed | %.0fs"
                  % (s, c + 1, n_chunks, got, time.time() - t0), flush=True)
        if got != rows:
            raise SystemExit("shard %d copied %d rows, expected %d" % (s, got, rows))
        row += rows
    vols.flush()
    np.save(os.path.join(a.out, "all_masks.npy"), masks)
    np.save(os.path.join(a.out, "all_ids.npy"), np.array(ids))

    # ---- verify, loudly. This is the check whose absence cost three weeks. ----
    print("\n[verify] re-opening from disk", flush=True)
    v = np.load(vols_p, mmap_mode="r")
    mm = np.load(os.path.join(a.out, "all_masks.npy"))
    ii = np.load(os.path.join(a.out, "all_ids.npy"), allow_pickle=True).astype(str)
    assert v.shape == (N_STUDIES, MAXS, IMG, IMG), "vols %r" % (v.shape,)
    assert mm.shape == (N_STUDIES, MAXS), "masks %r" % (mm.shape,)
    assert len(ii) == N_STUDIES and len(set(ii)) == N_STUDIES, "ids not %d unique" % N_STUDIES
    val = mm.mean(0)
    bounds, acc = [], 0
    for b in SLOT_BUDGETS:
        bounds.append((acc, acc + b)); acc += b
    print("[verify] per-slice validity by slot (expect a step at each boundary):")
    for si, (lo, hi) in enumerate(bounds):
        print("   slot %d  slices %2d-%2d  mean validity %.3f"
              % (si, lo, hi - 1, val[lo:hi].mean()))
    # Slot 3 (Coronal non-fluid) sits in only ~74% of studies. That asymmetry is the
    # fingerprint proving the slot boundaries are where we think they are.
    s3 = val[bounds[3][0]:bounds[3][1]].mean()
    if not (0.60 <= s3 <= 0.90):
        print("   WARNING slot 3 validity %.3f is outside the expected 0.60-0.90 for Coronal "
              "non-fluid; check the slot budgets before training." % s3)
    print("\nCORPUS OK  %r  | %d unique ids | %.2f GB | %.1f min"
          % (v.shape, len(ii), v.nbytes / 1e9, (time.time() - t0) / 60), flush=True)
    print("Point the trainer at it:  export KNEE_DATA_ROOT=%s" % os.path.abspath(a.out), flush=True)
    print("finetune_box.py will now pass its geometry gate (depth 96 == kernel MAXS).", flush=True)


if __name__ == "__main__":
    main()
