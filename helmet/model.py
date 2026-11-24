"""Faster R-CNN detectors with transfer learning, and checkpoint save / load.

Transfer learning: the detector is initialised from COCO-pretrained weights (backbone,
FPN, RPN and box head are all reused) and only the final classification / box-regression
layer is replaced so it predicts ``background + data.classes``. The early backbone stages
can be frozen with ``trainable_backbone_layers``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

ARCHITECTURES = {
    # name: (builder, COCO detection weights, ImageNet backbone weights)
    "fasterrcnn_mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights, MobileNet_V3_Large_Weights),
    "fasterrcnn_mobilenet_v3_large_320_fpn": (
        fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        MobileNet_V3_Large_Weights),
    "fasterrcnn_resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, ResNet50_Weights),
    "fasterrcnn_resnet50_fpn_v2": (
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, ResNet50_Weights),
}
MODEL_KWARGS = ("min_size", "max_size", "box_score_thresh", "box_nms_thresh", "box_detections_per_img")


def build_detector(arch: str, num_classes: Optional[int] = None, pretrained: str = "coco",
                   trainable_backbone_layers: Optional[int] = None, **kwargs) -> nn.Module:
    """Build a Faster R-CNN.

    pretrained: "coco" (full detector, transfer learning), "imagenet" (backbone only) or "none".
    num_classes: classes *including* background; None keeps the original 91 COCO classes.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    if pretrained not in ("coco", "imagenet", "none"):
        raise ValueError("pretrained must be 'coco', 'imagenet' or 'none'")
    builder, det_weights, backbone_weights = ARCHITECTURES[arch]
    weights = det_weights.DEFAULT if pretrained == "coco" else None
    weights_backbone = backbone_weights.DEFAULT if pretrained == "imagenet" else None
    if weights is None and weights_backbone is None:
        trainable_backbone_layers = None  # nothing pretrained -> everything is trained
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if pretrained == "none" and num_classes is not None:
        return builder(weights=None, weights_backbone=None, num_classes=num_classes, **kwargs)

    model = builder(weights=weights, weights_backbone=weights_backbone,
                    trainable_backbone_layers=trainable_backbone_layers, **kwargs)
    if num_classes is not None:  # transfer learning: new head for background + our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_helmet_model(cfg: dict, pretrained: Optional[str] = None) -> nn.Module:
    m = cfg["model"]
    return build_detector(
        m["arch"],
        num_classes=len(cfg["data"]["classes"]) + 1,
        pretrained=pretrained if pretrained is not None else m.get("pretrained", "coco"),
        trainable_backbone_layers=m.get("trainable_backbone_layers", 3),
        **{k: m.get(k) for k in MODEL_KWARGS},
    )


# --------------------------------------------------------------------------- checkpoints
def _frozen_bn_spec(model: nn.Module) -> Dict[str, float]:
    return {name: float(mod.eps) for name, mod in model.named_modules() if isinstance(mod, FrozenBatchNorm2d)}


def _apply_frozen_bn(model: nn.Module, spec: Dict[str, float]) -> None:
    """Pretrained torchvision backbones use FrozenBatchNorm2d, but a model built without
    weights uses BatchNorm2d. Re-create the exact layer types so the state dict matches
    and inference is numerically identical to the trained model."""
    for name, eps in spec.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        if isinstance(old, FrozenBatchNorm2d):
            continue
        num_features = old.num_features if hasattr(old, "num_features") else old.weight.shape[0]
        setattr(parent, child, FrozenBatchNorm2d(num_features, eps=eps))


def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict, epoch: int, metrics: dict,
                    optimizer=None, scheduler=None, best_score: Optional[float] = None,
                    best_tiebreak: Optional[float] = None) -> None:
    from .utils import to_python

    ckpt = {
        "model": model.state_dict(),
        "arch": cfg["model"]["arch"],
        "model_kwargs": {k: cfg["model"].get(k) for k in MODEL_KWARGS},
        "class_names": list(cfg["data"]["classes"]),
        "frozen_bn": _frozen_bn_spec(model),
        "config": cfg,
        "epoch": epoch,
        "metrics": to_python(metrics),
        "best_score": best_score,
        "best_tiebreak": best_tiebreak,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic: never leaves a half-written checkpoint


def load_checkpoint(path: str | Path) -> dict:
    # weights_only=False: the checkpoint also stores the run config (plain dicts/lists).
    # Only load checkpoints you created or trust.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_helmet_model(path: str | Path, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = load_checkpoint(path)
    model = build_detector(ckpt["arch"], num_classes=len(ckpt["class_names"]) + 1, pretrained="none",
                           **(ckpt.get("model_kwargs") or {}))
    _apply_frozen_bn(model, ckpt.get("frozen_bn", {}))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt
