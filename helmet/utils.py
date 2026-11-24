"""Small shared helpers: devices, seeding, collation, JSON-safe conversion."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

def select_device(name: str = "auto") -> torch.device:
    """auto = CUDA if available, else CPU.

    Apple-Silicon MPS is deliberately *not* auto-selected: with torch 2.8 / torchvision 0.23
    Faster R-CNN training on MPS produced exploding losses and hangs, and inference was
    slower than on the CPU (several detection ops fall back to the CPU). It can still be
    requested explicitly with ``--device mps``.
    """
    if name != "auto":
        if name == "mps":
            log.warning("MPS is experimental for torchvision detection models; use --device cpu if "
                        "losses explode or training stalls.")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so that wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own numpy/random stream (augmentations use them)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    """Detection batches are lists: images have different sizes and box counts."""
    return tuple(zip(*batch))


def to_python(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays into plain Python types."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_python(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_python(obj.detach().cpu().tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if np.isnan(value) else value
    return obj


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_python(data), f, indent=2)


def next_run_dir(base: str | Path) -> Path:
    """runs/exp -> runs/exp, runs/exp2, runs/exp3 ... (never overwrites a previous run)."""
    base = Path(base)
    if not base.exists():
        return base
    i = 2
    while Path(f"{base}{i}").exists():
        i += 1
    return Path(f"{base}{i}")
