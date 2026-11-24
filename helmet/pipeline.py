"""End-to-end per-frame inference: helmet/no-helmet heads + riders + tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .model import load_helmet_model
from .riders import RiderDetector, associate_heads
from .tracker import UNKNOWN, IoUTracker
from .utils import synchronize


@dataclass
class Detection:
    box: np.ndarray
    score: float
    label: int
    name: str


@dataclass
class Rider:
    box: np.ndarray
    score: float
    status: str  # a class name (e.g. "helmet" / "no_helmet") or "unknown"
    head: Optional[Detection] = None
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    heads: List[Detection] = field(default_factory=list)
    riders: List[Rider] = field(default_factory=list)
    enhanced: bool = False
    inference_ms: float = 0.0


def enhance_low_light(frame_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the lightness channel: lifts dark regions without blowing out highlights."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class HelmetPipeline:
    def __init__(self, weights: str, device: torch.device, conf_threshold: float = 0.5,
                 cross_class_nms_iou: float = 0.5, riders: Optional[dict] = None,
                 tracking: Optional[dict] = None, low_light: Optional[dict] = None,
                 violation_class: str = "no_helmet"):
        self.device = device
        self.model, ckpt = load_helmet_model(weights, device)
        self.class_names: List[str] = ckpt["class_names"]
        self.conf_threshold = conf_threshold
        self.cross_class_nms_iou = cross_class_nms_iou
        self.violation_class = violation_class

        riders = riders or {}
        self.rider_detector = None
        self.head_min_containment = riders.get("head_min_containment", 0.5)
        self.head_upper_fraction = riders.get("head_upper_fraction", 0.5)
        if riders.get("enabled", False):
            self.rider_detector = RiderDetector(
                device, arch=riders.get("arch", "fasterrcnn_mobilenet_v3_large_fpn"),
                person_conf=riders.get("person_conf", 0.5),
                motorcycle_conf=riders.get("motorcycle_conf", 0.4),
                min_x_overlap=riders.get("min_x_overlap", 0.5),
                min_bottom_depth=riders.get("min_bottom_depth", 0.5),
                min_height_ratio=riders.get("min_height_ratio", 0.8))

        tracking = tracking or {}
        self.tracker = None
        if tracking.get("enabled", False) and self.rider_detector is not None:
            self.tracker = IoUTracker(tracking.get("iou_threshold", 0.3), tracking.get("max_missed", 15),
                                      tracking.get("history", 15), tracking.get("min_hits_for_violation", 3),
                                      violation_status=violation_class)

        low_light = low_light or {}
        self.enhance = bool(low_light.get("enhance", False))
        self.brightness_threshold = float(low_light.get("brightness_threshold", 60))

    def reset_tracking(self) -> None:
        if self.tracker is not None:
            self.tracker = IoUTracker(self.tracker.iou_threshold, self.tracker.max_missed, self.tracker.history,
                                      self.tracker.min_hits_for_violation, self.tracker.violation_status)

    @property
    def violation_count(self) -> Optional[int]:
        return len(self.tracker.violators) if self.tracker is not None else None

    def _to_tensor(self, frame_bgr: np.ndarray):
        enhanced = False
        if self.enhance and cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean() < self.brightness_threshold:
            frame_bgr = enhance_low_light(frame_bgr)
            enhanced = True
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        return tensor, enhanced

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        synchronize(self.device)
        start = time.perf_counter()
        tensor, enhanced = self._to_tensor(frame_bgr)

        out = self.model([tensor.to(self.device)])[0]
        boxes, scores, labels = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
        keep = scores >= self.conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if len(boxes) and self.cross_class_nms_iou < 1.0:
            # a head is either helmet or no_helmet, never both: keep the more confident one
            keep = nms(boxes, scores, self.cross_class_nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        heads = [Detection(b.numpy(), float(s), int(l), self.class_names[int(l) - 1])
                 for b, s, l in zip(boxes, scores, labels)]

        riders: List[Rider] = []
        if self.rider_detector is not None:
            r_boxes, r_scores, _ = self.rider_detector(tensor)
            head_boxes = np.array([h.box for h in heads]).reshape(-1, 4)
            assignment = associate_heads(r_boxes, head_boxes, self.head_min_containment, self.head_upper_fraction)
            for ri, (rb, rs) in enumerate(zip(r_boxes, r_scores)):
                head = heads[assignment[ri]] if ri in assignment else None
                riders.append(Rider(rb, float(rs), head.name if head else UNKNOWN, head))
            if self.tracker is not None:
                tracked = self.tracker.update([r.box for r in riders], [r.status for r in riders])
                for rider, (track_id, status) in zip(riders, tracked):
                    rider.track_id, rider.status = track_id, status

        synchronize(self.device)
        return FrameResult(heads, riders, enhanced, (time.perf_counter() - start) * 1000)
