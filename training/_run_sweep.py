#!/usr/bin/env python3
"""Run a Vast-written knee script on Dread's local box under the GPU throttle.

The scripts under knee/ assume a rented Linux box with the card to themselves. This
wrapper installs the VRAM cap, duty-cycle pacing and Windows DataLoader safety first,
then hands over to the script unmodified, so there is one copy of each script rather
than a local fork that drifts.

    python _run_sweep.py <script.py> [args...]
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import _gpu_throttle
    _gpu_throttle.install()

    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]

    if os.environ.get("SLICE_DILATE"):
        import finetune_box as F
        F.SLICE_DILATE = int(os.environ["SLICE_DILATE"])
        print("[wrapper] SLICE_DILATE = %d" % F.SLICE_DILATE, flush=True)

    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
