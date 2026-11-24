"""Fine-tune a COCO-pretrained Faster R-CNN for helmet / no-helmet detection.

    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --set model.arch=fasterrcnn_resnet50_fpn_v2 train.epochs=40
    python train.py --resume runs/exp/last.pt
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import helmet  # noqa: F401  (sets MPS fallback before torch import)
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from helmet.config import apply_overrides, load_config, validate_config
from helmet.dataset import build_dataset, index_dataset
from helmet.engine import (build_optimizer, evaluate_predictions, predict, train_one_epoch,
                           warmup_cosine_lambda)
from helmet.metrics import format_metrics_table
from helmet.model import build_helmet_model, load_checkpoint, save_checkpoint
from helmet.plots import plot_training_history
from helmet.transforms import build_train_transforms
from helmet.utils import collate_fn, next_run_dir, save_json, seed_everything, select_device, worker_init_fn

log = logging.getLogger("train")


def make_loaders(cfg, records):
    t = cfg["train"]
    train_ds = build_dataset(cfg, "train", build_train_transforms(cfg.get("augment")), records)
    val_ds = build_dataset(cfg, "val", None, records)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train and val splits must both contain images")
    sampler = None
    if t.get("balanced_sampling"):
        sampler = WeightedRandomSampler(train_ds.sampling_weights(), num_samples=len(train_ds), replacement=True)
    common = dict(num_workers=t.get("num_workers", 2), collate_fn=collate_fn, worker_init_fn=worker_init_fn,
                  persistent_workers=t.get("num_workers", 2) > 0)
    g = torch.Generator()
    g.manual_seed(t.get("seed", 42))
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=sampler is None, sampler=sampler,
                              drop_last=len(train_ds) > t["batch_size"], generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, **common)
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--resume", type=Path, help="continue training from a last.pt checkpoint")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_checkpoint(args.resume)
        cfg = apply_overrides(resume_ckpt["config"], args.set)
        validate_config(cfg)
        out_dir = args.resume.parent
    else:
        cfg = load_config(args.config, args.set)
        out_dir = next_run_dir(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t = cfg["train"]
    seed_everything(t.get("seed", 42))
    device = select_device(args.device)
    log.info("Device: %s | output: %s", device, out_dir)

    records = index_dataset(cfg["data"])
    train_loader, val_loader = make_loaders(cfg, records)
    log.info("Train images: %d | val images: %d | classes: %s", len(train_loader.dataset),
             len(val_loader.dataset), cfg["data"]["classes"])

    # Built exactly as in the original run (same frozen layers / FrozenBatchNorm), so that
    # on --resume the weights and the optimizer state line up with the parameters.
    model = build_helmet_model(cfg)
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s (%.1fM trainable params)", cfg["model"]["arch"], n_train / 1e6)

    optimizer = build_optimizer(model, t)
    total_iters = t["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(total_iters, t.get("warmup_iters", 300), min_lr_ratio=t.get("min_lr_ratio", 0.01)))

    start_epoch, history, bad_epochs = 1, [], 0
    best_score, best_tiebreak = -1.0, -1.0  # primary metric, then mAP@0.5:0.95 to break ties
    history_path = out_dir / "history.csv"
    if resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_score = resume_ckpt.get("best_score") or -1.0
        best_tiebreak = resume_ckpt.get("best_tiebreak") or -1.0
        if history_path.exists():
            with open(history_path) as f:
                history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
            history = [h for h in history if h["epoch"] < start_epoch]
            for h in history:
                h["epoch"] = int(h["epoch"])
        log.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    best_key = t.get("best_metric", "map50")
    conf = t.get("eval_conf_threshold", 0.5)
    patience = t.get("early_stopping_patience", 0)
    for epoch in range(start_epoch, t["epochs"] + 1):
        losses = train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch,
                                 t.get("grad_clip", 0.0), t.get("amp", False))
        ev = evaluate_predictions(predict(model, val_loader, device, desc="validate"), cfg["data"]["classes"])
        metrics = ev.compute(conf)
        score = metrics[best_key]
        score = -1.0 if score is None or score != score else score

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in losses.items()},
               **{f"val_{k}": metrics[k] for k in ("map50", "map50_95", "precision", "recall", "f1")},
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        log.info("epoch %d/%d | loss %.4f | val mAP50 %.4f | mAP50-95 %.4f | P %.3f R %.3f F1 %.3f",
                 epoch, t["epochs"], losses["loss"], metrics["map50"], metrics["map50_95"],
                 metrics["precision"], metrics["recall"], metrics["f1"])

        tiebreak = metrics["map50_95"] if metrics["map50_95"] == metrics["map50_95"] else -1.0
        improved = score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and tiebreak > best_tiebreak)
        if improved:
            best_score, best_tiebreak, bad_epochs = score, tiebreak, 0
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch, metrics, best_score=best_score,
                            best_tiebreak=best_tiebreak)
            save_json(metrics, out_dir / "val_metrics_best.json")
            log.info("  new best %s = %.4f -> %s", best_key, best_score, out_dir / "best.pt")
        else:
            bad_epochs += 1
        save_checkpoint(out_dir / "last.pt", model, cfg, epoch, metrics, optimizer, scheduler, best_score,
                        best_tiebreak=best_tiebreak)
        plot_training_history(history, out_dir / "training_curves.png")
        if patience and bad_epochs >= patience:
            log.info("Early stopping: no improvement in %s for %d epochs", best_key, patience)
            break

    if (out_dir / "best.pt").exists():
        best = load_checkpoint(out_dir / "best.pt")
        log.info("Best epoch %d (%s = %.4f). Validation metrics:\n%s", best["epoch"], best_key, best_score,
                 format_metrics_table(best["metrics"]))
    log.info("Done. Evaluate with: python evaluate.py --weights %s --stress", out_dir / "best.pt")


if __name__ == "__main__":
    main()
