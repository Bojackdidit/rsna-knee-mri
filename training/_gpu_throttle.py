#!/usr/bin/env python3
"""Keep a local GPU job from making Dread's desktop unusable.

The 4070 Ti drives his display. Two things make the box unusable: exhausting the 12 GB
(the compositor starts thrashing) and holding the SM at 100% for hours. Both are capped
here in code rather than by anyone watching a terminal.

  VRAM   torch.cuda.set_per_process_memory_fraction - a hard allocator ceiling.
  DUTY   after each batch, sleep in proportion to how long that batch took, so our
         share of the SM stays near DUTY. Measured per batch, so it adapts to K.
  TEMP   checked every few seconds; over the limit we add extra sleep until it drops.

Nothing here is specific to a script - it monkeypatches DataLoader, so the caller runs
unmodified. Env overrides: KNEE_VRAM_FRAC, KNEE_DUTY, KNEE_TEMP_LIMIT.
"""
import os
import subprocess
import time

import torch
from torch.utils.data import DataLoader

_orig_iter = DataLoader.__iter__
_installed = False


def _gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL)
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def _lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "BELOW_NORMAL"
    except Exception:
        return "unchanged"


def install_vram_only(frac=None, verbose=True):
    """Cap VRAM but leave pacing off - for benchmarks, which must measure raw capability."""
    frac = float(os.environ.get("KNEE_VRAM_FRAC", frac if frac is not None else 0.65))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if verbose:
            print("[throttle] VRAM cap %.0f%% = %.1f GB of %.1f | pacing OFF (benchmark)"
                  % (frac * 100, frac * total, total), flush=True)


def install(frac=None, duty=None, temp_limit=None, verbose=True):
    """Cap VRAM, pace every DataLoader, and back off when the card gets hot."""
    global _installed
    frac = float(os.environ.get("KNEE_VRAM_FRAC", frac if frac is not None else 0.65))
    duty = float(os.environ.get("KNEE_DUTY", duty if duty is not None else 0.55))
    temp_limit = int(os.environ.get("KNEE_TEMP_LIMIT", temp_limit if temp_limit is not None else 75))

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        prio = _lower_priority()
        if verbose:
            print("[throttle] %s | VRAM cap %.0f%% = %.1f GB of %.1f | duty %.0f%% | temp limit %dC | prio %s"
                  % (torch.cuda.get_device_name(0), frac * 100, frac * total, total,
                     duty * 100, temp_limit, prio), flush=True)
    elif verbose:
        print("[throttle] no CUDA device - pacing disabled", flush=True)
        return

    if _installed:
        return
    _installed = True

    # These scripts were written for the Vast boxes, where DataLoader workers fork.
    # Windows spawns instead: each worker re-imports the main module, and a script with
    # no __main__ guard re-runs itself. Worse, the respawn here came back on the CPU-only
    # system Python rather than this venv, so it hung holding 1.4 GB. Clamp workers to 0
    # unless someone opts in explicitly, so the unmodified scripts are safe on this box.
    workers = int(os.environ.get("KNEE_WORKERS", 0))
    _orig_init = DataLoader.__init__

    def clamped_init(self, *a, **kw):
        if kw.get("num_workers", 0) != workers:
            kw["num_workers"] = workers
            kw.pop("prefetch_factor", None)
            if workers == 0:
                kw.pop("persistent_workers", None)
        return _orig_init(self, *a, **kw)

    DataLoader.__init__ = clamped_init
    if verbose:
        print("[throttle] DataLoader num_workers forced to %d (Windows spawn safety)" % workers, flush=True)

    state = {"extra": 0.0, "checked": 0.0, "slept": 0.0, "busy": 0.0}

    def paced(self):
        for batch in _orig_iter(self):
            t0 = time.perf_counter()
            yield batch
            dt = time.perf_counter() - t0
            state["busy"] += dt

            now = time.time()
            if now - state["checked"] > 5.0:
                state["checked"] = now
                t = _gpu_temp()
                if t is not None:
                    if t > temp_limit:
                        state["extra"] = min(state["extra"] + 0.05, 1.0)
                    elif t < temp_limit - 5 and state["extra"] > 0:
                        state["extra"] = max(state["extra"] - 0.05, 0.0)

            nap = dt * (1.0 / duty - 1.0) + state["extra"]
            if nap > 0:
                state["slept"] += nap
                time.sleep(nap)

    DataLoader.__iter__ = paced
    return state
