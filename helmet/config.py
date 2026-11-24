"""YAML configuration loading with ``key.sub=value`` command-line overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

SUPPORTED_FORMATS = ("voc", "yolo")


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: Optional[Iterable[str]]) -> dict:
    """Apply overrides like ``train.lr=0.005`` (values are parsed as YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def validate_config(cfg: dict) -> None:
    required = {
        "data": ["root", "format", "classes", "splits_dir"],
        "model": ["arch"],
        "train": ["epochs", "batch_size", "lr", "output_dir"],
    }
    for section, keys in required.items():
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Config is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config is missing '{section}.{key}'")

    data = cfg["data"]
    if data["format"] not in SUPPORTED_FORMATS:
        raise ValueError(f"data.format must be one of {SUPPORTED_FORMATS}, got {data['format']!r}")
    classes = data["classes"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("data.classes must be a non-empty list of unique names")
    if data["format"] == "yolo" and not data.get("yolo_names"):
        raise ValueError("data.yolo_names (class names in YOLO id order) is required for YOLO datasets")
