"""Dataset indexing and loading for helmet / no-helmet detection.

Two annotation layouts are supported:

* Pascal VOC  : <root>/<images_dir>/*.jpg|png  and  <root>/<annotations_dir>/*.xml
* YOLO (txt)  : <root>/<images_dir>/*.jpg|png  and  <root>/<labels_dir>/*.txt
                (``class cx cy w h`` normalised, or polygon rows from segmentation exports)

Raw class names from the annotation files are mapped onto the project's canonical
classes (``data.classes``) through ``data.aliases`` so that datasets that call the
classes "With Helmet" / "Without Helmet", "helmet" / "head", etc. all work.
Objects whose class is not mapped (e.g. "number plate") are ignored.

Label convention used everywhere in the project: 0 = background, 1..C = data.classes.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PER_BOX_KEYS = ("boxes", "labels", "occluded", "truncated")


def normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


class LabelMapper:
    """Maps raw annotation class names to project label ids (1..C)."""

    def __init__(self, classes: Sequence[str], aliases: Optional[Dict[str, List[str]]] = None):
        self.classes = list(classes)
        self._map: Dict[str, int] = {}
        for idx, cls in enumerate(self.classes, start=1):
            for raw in [cls, *((aliases or {}).get(cls) or [])]:
                key = normalize_name(raw)
                if key in self._map and self._map[key] != idx:
                    raise ValueError(f"Alias {raw!r} is mapped to two different classes")
                self._map[key] = idx

    def __call__(self, raw_name: str) -> Optional[int]:
        return self._map.get(normalize_name(raw_name))


# --------------------------------------------------------------------------- parsing
def _flag(obj: ET.Element, tag: str) -> bool:
    text = obj.findtext(tag)
    if text is None or not text.strip():
        return False
    try:
        return bool(int(float(text)))
    except ValueError:
        return text.strip().lower() in ("true", "yes")


def parse_voc_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    objects = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            box = [float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            log.warning("Skipping malformed box in %s", xml_path)
            continue
        objects.append(
            {
                "name": (obj.findtext("name") or "").strip(),
                "box": box,
                "occluded": _flag(obj, "occluded"),
                "truncated": _flag(obj, "truncated"),
            }
        )
    return {"filename": root.findtext("filename"), "objects": objects}


def parse_yolo_txt(txt_path: Path) -> List[tuple]:
    """Returns [(class_id, x1, y1, x2, y2)] in normalised [0, 1] coordinates."""
    rows = []
    for line_no, line in enumerate(txt_path.read_text().splitlines(), start=1):
        vals = line.split()
        if not vals:
            continue
        try:
            cls_id = int(float(vals[0]))
            coords = [float(v) for v in vals[1:]]
        except ValueError:
            log.warning("Skipping malformed line %d in %s", line_no, txt_path)
            continue
        if len(coords) == 4:
            cx, cy, w, h = coords
            rows.append((cls_id, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        elif len(coords) >= 6 and len(coords) % 2 == 0:  # polygon (segmentation export)
            xs, ys = coords[0::2], coords[1::2]
            rows.append((cls_id, min(xs), min(ys), max(xs), max(ys)))
        else:
            log.warning("Skipping line %d in %s: unexpected number of values", line_no, txt_path)
    return rows


def _index_images(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    index: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in index:
                log.warning("Duplicate image stem %r, using %s", path.stem, index[path.stem].name)
                continue
            index[path.stem] = path
    return index


def _empty_record(stem: str, image_path: Path, normalized: bool) -> dict:
    return {
        "stem": stem,
        "image_path": str(image_path),
        "boxes": np.zeros((0, 4), np.float32),
        "labels": np.zeros((0,), np.int64),
        "occluded": np.zeros((0,), bool),
        "truncated": np.zeros((0,), bool),
        "normalized": normalized,
    }


def index_dataset(data_cfg: dict) -> Dict[str, dict]:
    """Scan the dataset folder and return {image stem: annotation record}."""
    root = Path(data_cfg["root"])
    images = _index_images(root / data_cfg.get("images_dir", "images"))
    mapper = LabelMapper(data_cfg["classes"], data_cfg.get("aliases"))
    fmt = data_cfg["format"]
    ignored: Counter = Counter()
    records: Dict[str, dict] = {}

    if fmt == "voc":
        ann_dir = root / data_cfg.get("annotations_dir", "annotations")
        ann_files = sorted(ann_dir.glob("*.xml"))
    else:
        ann_dir = root / data_cfg.get("labels_dir", "labels")
        ann_files = sorted(ann_dir.glob("*.txt"))
        yolo_names = list(data_cfg["yolo_names"])
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    missing_images = 0
    for ann_path in ann_files:
        image_path = images.get(ann_path.stem)
        if image_path is None:
            missing_images += 1
            continue
        boxes, labels, occl, trunc = [], [], [], []
        if fmt == "voc":
            for obj in parse_voc_xml(ann_path)["objects"]:
                label = mapper(obj["name"])
                if label is None:
                    ignored[obj["name"]] += 1
                    continue
                boxes.append(obj["box"])
                labels.append(label)
                occl.append(obj["occluded"])
                trunc.append(obj["truncated"])
        else:
            for cls_id, *box in parse_yolo_txt(ann_path):
                if not 0 <= cls_id < len(yolo_names):
                    ignored[f"<yolo id {cls_id}>"] += 1
                    continue
                label = mapper(yolo_names[cls_id])
                if label is None:
                    ignored[yolo_names[cls_id]] += 1
                    continue
                boxes.append(box)
                labels.append(label)
                occl.append(False)
                trunc.append(False)
        rec = _empty_record(ann_path.stem, image_path, normalized=(fmt == "yolo"))
        if boxes:
            rec["boxes"] = np.asarray(boxes, np.float32).reshape(-1, 4)
            rec["labels"] = np.asarray(labels, np.int64)
            rec["occluded"] = np.asarray(occl, bool)
            rec["truncated"] = np.asarray(trunc, bool)
        records[ann_path.stem] = rec

    if missing_images:
        log.warning("%d annotation files have no matching image and were skipped", missing_images)
    if ignored:
        log.info("Ignored objects with unmapped classes: %s", dict(ignored))
    if not records:
        raise RuntimeError(f"No annotated images found under {root}")
    return records


# --------------------------------------------------------------------------- splits
def read_split(splits_dir: str | Path, split: str) -> List[str]:
    path = Path(splits_dir) / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found. Run prepare_data.py first.")
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def filter_target(target: dict, keep: np.ndarray) -> dict:
    """Apply a boolean/index mask to every per-box array of a target dict."""
    return {k: (v[keep] if k in PER_BOX_KEYS else v) for k, v in target.items()}


# --------------------------------------------------------------------------- dataset
class HelmetDataset(Dataset):
    """Returns ``(image_tensor[3,H,W] float 0..1, target)`` in torchvision detection format."""

    def __init__(
        self,
        records: Sequence[dict],
        class_names: Sequence[str],
        transforms: Optional[Callable] = None,
        min_box_size: float = 2.0,
    ):
        self.records = list(records)
        self.class_names = list(class_names)
        self.transforms = transforms
        self.min_box_size = float(min_box_size)

    def __len__(self) -> int:
        return len(self.records)

    def load(self, idx: int):
        """Raw RGB uint8 image and cleaned numpy target (absolute xyxy pixel boxes)."""
        rec = self.records[idx]
        image = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image {rec['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        boxes = rec["boxes"].astype(np.float32).copy()
        if rec["normalized"]:
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        target = {
            "boxes": boxes,
            "labels": rec["labels"].copy(),
            "occluded": rec["occluded"].copy(),
            "truncated": rec["truncated"].copy(),
        }
        keep = ((boxes[:, 2] - boxes[:, 0]) >= self.min_box_size) & (
            (boxes[:, 3] - boxes[:, 1]) >= self.min_box_size
        )
        return image, filter_target(target, keep)

    def __getitem__(self, idx: int):
        image, target = self.load(idx)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return to_tensor_sample(image, target, idx)

    def label_counts(self) -> Counter:
        counts: Counter = Counter()
        for rec in self.records:
            counts.update(int(l) for l in rec["labels"])
        return counts

    def sampling_weights(self) -> List[float]:
        """Per-image weights that over-sample images containing rare classes."""
        counts = self.label_counts()
        weights = []
        for rec in self.records:
            labels = set(int(l) for l in rec["labels"])
            weights.append(max((1.0 / counts[l] for l in labels), default=1.0 / max(len(self.records), 1)))
        return weights


class DeterministicTransformDataset(Dataset):
    """Applies ``fn(image, target, rng)`` with a per-image seeded RNG (used for the
    reproducible robustness stress tests at evaluation time)."""

    def __init__(self, base: HelmetDataset, fn: Callable, seed: int = 0):
        self.base, self.fn, self.seed = base, fn, seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, target = self.base.load(idx)
        image, target = self.fn(image, target, np.random.default_rng(self.seed + idx))
        return to_tensor_sample(image, target, idx)


def to_tensor_sample(image: np.ndarray, target: dict, idx: int):
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    out = {
        "boxes": torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.as_tensor(target["labels"], dtype=torch.int64).reshape(-1),
        "image_id": torch.tensor(idx, dtype=torch.int64),
    }
    return tensor, out


def build_dataset(cfg: dict, split: str, transforms: Optional[Callable] = None,
                  records: Optional[Dict[str, dict]] = None) -> HelmetDataset:
    data = cfg["data"]
    records = records if records is not None else index_dataset(data)
    stems = read_split(data["splits_dir"], split)
    missing = [s for s in stems if s not in records]
    if missing:
        raise KeyError(f"{len(missing)} images listed in the {split} split are missing, e.g. {missing[:3]}")
    return HelmetDataset([records[s] for s in stems], data["classes"], transforms,
                         data.get("min_box_size", 2))
