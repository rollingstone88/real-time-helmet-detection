"""Create stratified train / val / test splits.

Flat dataset (e.g. Kaggle "Helmet Detection", Pascal VOC):
    python prepare_data.py --config configs/default.yaml

Pre-split YOLO export (e.g. Roboflow: <dir>/{train,valid,test}/{images,labels}):
    python prepare_data.py --config configs/yolo.yaml --from-presplit path/to/export
    -> copies files into data.root/{images,labels} and keeps the original split.
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import helmet  # noqa: F401
from helmet.config import load_config
from helmet.dataset import IMAGE_EXTENSIONS, index_dataset

log = logging.getLogger("prepare_data")
PRESPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def stratified_split(records: dict, ratios, seed: int):
    """Split image stems so each set has a similar mix of classes (groups by the set of
    classes present in the image, e.g. {helmet}, {no_helmet}, {helmet, no_helmet})."""
    groups = defaultdict(list)
    for stem, rec in sorted(records.items()):
        groups[tuple(sorted(set(int(l) for l in rec["labels"])))].append(stem)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        stems = groups[key]
        rng.shuffle(stems)
        n = len(stems)
        n_val, n_test = int(round(n * ratios[1])), int(round(n * ratios[2]))
        n_val, n_test = min(n_val, n), min(n_test, n - min(n_val, n))
        splits["val"] += stems[:n_val]
        splits["test"] += stems[n_val:n_val + n_test]
        splits["train"] += stems[n_val + n_test:]
    for v in splits.values():
        rng.shuffle(v)
    return splits


def import_presplit(src: Path, data_cfg: dict):
    root = Path(data_cfg["root"])
    img_out = root / data_cfg.get("images_dir", "images")
    lbl_out = root / data_cfg.get("labels_dir", "labels")
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for sub, split in PRESPLIT_NAMES.items():
        img_dir, lbl_dir = src / sub / "images", src / sub / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                (lbl_out / f"{img.stem}.txt").write_text("")  # background image (no objects)
            else:
                shutil.copy2(lbl, lbl_out / lbl.name)
            shutil.copy2(img, img_out / img.name)
            splits[split].append(img.stem)
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-presplit", type=Path, help="YOLO export with train/valid/test folders")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config, args.set)
    if args.from_presplit:
        if cfg["data"]["format"] != "yolo":
            ap.error("--from-presplit expects a YOLO-format export; set data.format=yolo")
        splits = import_presplit(args.from_presplit, cfg["data"])
        records = index_dataset(cfg["data"])
        splits = {k: [s for s in v if s in records] for k, v in splits.items()}
    else:
        if not 0 <= args.val + args.test < 1:
            ap.error("--val + --test must be in [0, 1)")
        records = index_dataset(cfg["data"])
        splits = stratified_split(records, (1 - args.val - args.test, args.val, args.test), args.seed)

    out = Path(cfg["data"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    classes = cfg["data"]["classes"]
    for name, stems in splits.items():
        (out / f"{name}.txt").write_text("\n".join(stems) + ("\n" if stems else ""))
        counts = Counter()
        for s in stems:
            counts.update(int(l) for l in records[s]["labels"])
        per_class = ", ".join(f"{c}={counts[i + 1]}" for i, c in enumerate(classes))
        log.info("%-5s: %5d images | boxes: %s", name, len(stems), per_class)
    log.info("Split files written to %s", out)


if __name__ == "__main__":
    main()
