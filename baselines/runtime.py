"""Runtime helpers shared by classic baselines.

These helpers validate, seed, and clear the explicitly selected device so
independent benchmark processes can use different GPUs.
"""

import os
import random
import time

import numpy as np
import torch


def print_stage(method, phase, phase_total, message):
    """Print the current coarse phase before a potentially long operation."""
    print(f"    [{method}-stage {phase}/{phase_total}] {message}", flush=True)


def print_progress(method, label, completed, total, started):
    """Render a compact dependency-free progress bar with elapsed time/ETA."""
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    elapsed = time.perf_counter() - started
    eta = elapsed / completed * (total - completed) if completed else 0.0
    width = 20
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if completed == total else "\r"
    print(
        f"    [{method}-progress] {label} [{bar}] {completed}/{total} "
        f"({100.0 * completed / total:5.1f}%) "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
        end=end,
        flush=True,
    )


def select_device(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {dev}")
        torch.cuda.set_device(dev)
    return dev


def seed_for_device(seed, device=None):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.default_generator.manual_seed(seed)
    if device is not None:
        dev = select_device(device)
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def empty_device_cache(device):
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
