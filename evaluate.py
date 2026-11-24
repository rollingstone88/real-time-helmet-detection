"""Evaluate a trained model: precision, recall, F1, mAP@0.5, mAP@0.5:0.95, confusion
matrix, per-condition breakdown and robustness stress tests.

    python evaluate.py --weights runs/exp/best.pt                 # test split
    python evaluate.py --weights runs/exp/best.pt --stress        # + occlusion / lighting / blur tests

Breakdowns reported (when the subset is non-empty):
  lighting  : dark / normal / bright images (mean luminance)
  riders    : images with a single rider head vs. multiple riders
  occlusion : images whose annotations flag occluded / truncated objects (VOC datasets)
Stress tests re-run the test set with synthetic low light, over-exposure, 30% occlusion of
every head and motion blur, and report the drop versus clean images.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import helmet  # noqa: F401
import numpy as np
from torch.utils.data import DataLoader

from helmet.config import apply_overrides, load_config
from helmet.dataset import DeterministicTransformDataset, build_dataset
from helmet.engine import evaluate_predictions, mean_inference_ms, predict
from helmet.metrics import format_metrics_table
from helmet.model import load_helmet_model
from helmet.plots import plot_confusion_matrix, plot_f1_curve, plot_pr_curves
from helmet.transforms import STRESS_TESTS
from helmet.utils import collate_fn, save_json, select_device

log = logging.getLogger("evaluate")
DARK, BRIGHT = 70.0, 170.0  # mean luminance thresholds (0-255)
SUMMARY_KEYS = ("map50", "map50_95", "precision", "recall", "f1")


def summarize(results, class_names, conf):
    if not results:
        return None
    m = evaluate_predictions(results, class_names).compute(conf)
    return {"num_images": len(results), **{k: m[k] for k in SUMMARY_KEYS},
            "per_class_ap50": {c: v["ap50"] for c, v in m["per_class"].items()}}


def print_summary_table(title, rows):
    log.info("\n%s\n%-22s%7s%8s%9s%8s%8s%8s", title, "subset", "images", "mAP50", "mAP50-95", "P", "R", "F1")
    for name, s in rows.items():
        if s is None:
            continue
        log.info("%-22s%7d%8.3f%9.3f%8.3f%8.3f%8.3f", name, s["num_images"], _nan(s["map50"]),
                 _nan(s["map50_95"]), s["precision"], s["recall"], s["f1"])


def _nan(v):
    return float("nan") if v is None else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--config", help="optional config (e.g. different data paths); defaults to the training config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, help="confidence threshold for P/R/F1 (default: from config)")
    ap.add_argument("--stress", action="store_true", help="run robustness stress tests")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, help="output folder (default: <weights dir>/eval_<split>)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = select_device(args.device)
    model, ckpt = load_helmet_model(args.weights, device)
    cfg = load_config(args.config, args.set) if args.config else apply_overrides(ckpt["config"], args.set)
    class_names = ckpt["class_names"]
    if list(cfg["data"]["classes"]) != class_names:
        raise SystemExit(f"Config classes {cfg['data']['classes']} differ from the checkpoint's {class_names}")
    conf = args.conf if args.conf is not None else cfg["train"].get("eval_conf_threshold", 0.5)
    out = args.out or args.weights.parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(cfg, args.split)
    loader_kw = dict(batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    log.info("Evaluating %s on %d %s images (device %s, conf %.2f)", args.weights, len(ds), args.split, device, conf)

    results = predict(model, DataLoader(ds, **loader_kw), device, desc=args.split)
    ev = evaluate_predictions(results, class_names)
    metrics = ev.compute(conf)
    ms = mean_inference_ms(results)
    metrics["inference_ms_per_image"] = ms
    metrics["inference_fps"] = 1000.0 / ms if ms == ms and ms > 0 else None

    log.info("\nOverall (%s split)\n%s", args.split, format_metrics_table(metrics))
    log.info("mAP@0.75 %.3f | macro F1 %.3f | best F1 %.3f at conf %.2f | %.1f ms/image (%.1f FPS on %s)",
             _nan(metrics["map75"]), metrics["macro_f1"], metrics["best_f1"], metrics["best_f1_conf_threshold"],
             ms, metrics["inference_fps"] or 0, device)
    labels = metrics["confusion_labels"]
    log.info("\nConfusion matrix (rows = ground truth, cols = predicted)\n%14s%s", "",
             "".join(f"{l:>12}" for l in labels))
    for name, row in zip(labels, metrics["confusion_matrix"]):
        log.info("%14s%s", name, "".join(f"{v:>12d}" for v in row))

    # ---- breakdown by condition -------------------------------------------------------
    flags = [bool(ds.records[r["image_id"]]["occluded"].any() or ds.records[r["image_id"]]["truncated"].any())
             for r in results]
    subsets = {
        "all": results,
        "lighting: dark": [r for r in results if r["brightness"] < DARK],
        "lighting: normal": [r for r in results if DARK <= r["brightness"] < BRIGHT],
        "lighting: bright": [r for r in results if r["brightness"] >= BRIGHT],
        "riders: single": [r for r in results if len(r["gt"]["labels"]) == 1],
        "riders: multiple": [r for r in results if len(r["gt"]["labels"]) >= 2],
        "occluded/truncated": [r for r, f in zip(results, flags) if f],
    }
    breakdown = {k: summarize(v, class_names, conf) for k, v in subsets.items()}
    print_summary_table("Breakdown by condition", breakdown)

    # ---- robustness stress tests --------------------------------------------------------
    stress = {}
    if args.stress:
        clean = breakdown["all"]
        stress["clean"] = clean
        for name, fn in STRESS_TESTS.items():
            sds = DeterministicTransformDataset(ds, fn, seed=1234)
            s = summarize(predict(model, DataLoader(sds, **loader_kw), device, desc=name), class_names, conf)
            s["delta_map50"] = s["map50"] - clean["map50"] if s["map50"] is not None else None
            stress[name] = s
        print_summary_table("Robustness stress tests", stress)

    # ---- save -------------------------------------------------------------------------
    report = {"weights": str(args.weights), "split": args.split, "metrics": metrics,
              "breakdown": breakdown, "stress_tests": stress,
              "lighting_thresholds": {"dark_below": DARK, "bright_from": BRIGHT}}
    save_json(report, out / "report.json")
    plot_pr_curves(ev.pr_curves(), out / "pr_curve.png")
    plot_f1_curve(*ev.f1_curve(), out / "f1_curve.png")
    plot_confusion_matrix(metrics["confusion_matrix"], labels, out / "confusion_matrix.png")
    log.info("\nSaved report.json, pr_curve.png, f1_curve.png, confusion_matrix.png to %s", out)


if __name__ == "__main__":
    main()
