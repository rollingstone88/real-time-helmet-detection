"""Training and validation loops."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .metrics import DetectionEvaluator
from .utils import synchronize


def warmup_cosine_lambda(total_iters: int, warmup_iters: int, warmup_factor: float = 1e-3,
                         min_lr_ratio: float = 0.01):
    """LR multiplier: linear warm-up, then cosine decay to ``min_lr_ratio``."""
    warmup_iters = min(warmup_iters, max(total_iters - 1, 0))

    def fn(it: int) -> float:
        if it < warmup_iters:
            alpha = it / max(1, warmup_iters)
            return warmup_factor * (1 - alpha) + alpha
        progress = min(1.0, (it - warmup_iters) / max(1, total_iters - warmup_iters))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = train_cfg.get("optimizer", "sgd").lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=train_cfg["lr"], momentum=train_cfg.get("momentum", 0.9),
                               weight_decay=train_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 1e-4))
    raise ValueError(f"Unknown optimizer {name!r} (use 'sgd' or 'adamw')")


def train_one_epoch(model, optimizer, scheduler, loader, device, epoch: int, grad_clip: float = 0.0,
                    use_amp: bool = False) -> Dict[str, float]:
    model.train()
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sums: Dict[str, float] = defaultdict(float)
    n = 0
    bar = tqdm(loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
    for images, targets in bar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            raise FloatingPointError(
                f"Non-finite loss {loss.item()} ({ {k: v.item() for k, v in loss_dict.items()} }). "
                "Try a lower learning rate or check the annotations for broken boxes.")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        n += 1
        sums["loss"] += loss.item()
        for k, v in loss_dict.items():
            sums[k] += v.item()
        bar.set_postfix(loss=f"{sums['loss'] / n:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader, device, desc: str = "eval") -> List[dict]:
    """Run the model over a loader; returns per-image numpy predictions, ground truth,
    mean image brightness (0-255) and per-image inference time in ms."""
    model.eval()
    results = []
    for images, targets in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        dev_images = [img.to(device, non_blocking=True) for img in images]
        synchronize(device)
        start = time.perf_counter()
        outputs = model(dev_images)
        synchronize(device)
        ms = (time.perf_counter() - start) * 1000 / len(images)
        for img, out, tgt in zip(images, outputs, targets):
            luminance = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).mean().item() * 255
            results.append({
                "pred": {k: out[k].detach().cpu().numpy() for k in ("boxes", "scores", "labels")},
                "gt": {"boxes": tgt["boxes"].numpy(), "labels": tgt["labels"].numpy()},
                "image_id": int(tgt["image_id"]),
                "brightness": luminance,
                "time_ms": ms,
            })
    return results


def evaluate_predictions(results: Sequence[dict], class_names: Sequence[str]) -> DetectionEvaluator:
    ev = DetectionEvaluator(class_names)
    for r in results:
        ev.add(r["pred"], r["gt"])
    return ev


def mean_inference_ms(results: Sequence[dict], warmup: int = 3) -> float:
    times = [r["time_ms"] for r in results]
    times = times[warmup:] if len(times) > warmup else times
    return float(np.mean(times)) if times else float("nan")
