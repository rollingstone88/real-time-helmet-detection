"""Motorcycle-rider detection and head-to-rider association.

Riders are found with a COCO-pretrained detector: a *person* box is a rider when it sits
on a *motorcycle* box. Helmet / no-helmet head detections from the fine-tuned model are
then assigned to riders (one head per rider), which handles several riders on the same
or different motorcycles in one frame.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .model import build_detector

COCO_PERSON = 1
COCO_MOTORCYCLE = 4


class RiderDetector:
    def __init__(self, device: torch.device, arch: str = "fasterrcnn_mobilenet_v3_large_fpn",
                 person_conf: float = 0.5, motorcycle_conf: float = 0.4, min_x_overlap: float = 0.5,
                 min_bottom_depth: float = 0.5, min_height_ratio: float = 0.8):
        self.device = device
        self.person_conf = person_conf
        self.motorcycle_conf = motorcycle_conf
        self.min_x_overlap = min_x_overlap
        self.min_bottom_depth = min_bottom_depth
        self.min_height_ratio = min_height_ratio
        self.model = build_detector(arch, pretrained="coco",
                                    box_score_thresh=min(person_conf, motorcycle_conf)).to(device).eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """image: float tensor (3,H,W) in [0,1]. Returns rider boxes, rider scores, motorcycle boxes."""
        out = self.model([image.to(self.device)])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        persons = (labels == COCO_PERSON) & (scores >= self.person_conf)
        motos = (labels == COCO_MOTORCYCLE) & (scores >= self.motorcycle_conf)
        idx = find_riders(boxes[persons], boxes[motos], self.min_x_overlap, self.min_bottom_depth,
                          self.min_height_ratio)
        return boxes[persons][idx], scores[persons][idx], boxes[motos]


def is_riding(person: Sequence[float], moto: Sequence[float], min_x_overlap: float = 0.5,
              min_bottom_depth: float = 0.5, min_height_ratio: float = 0.8) -> bool:
    """Geometric test for "this person is sitting on this motorcycle":

    * the person overlaps the motorcycle horizontally (``min_x_overlap`` of the narrower box),
    * their upper body is above the motorcycle's top,
    * their lower body reaches at least ``min_bottom_depth`` of the way down the motorcycle
      box (pedestrians standing *behind* a bike end higher up, at their feet), and
    * they are at least ``min_height_ratio`` x the motorcycle's height (people further
      away behind the bike look smaller).
    The last two thresholds were set from real street photos: riders measured >= 0.55 depth
    and >= 0.9 height ratio, pedestrians behind a bike <= 0.46 and <= 0.73.
    """
    px1, py1, px2, py2 = person
    mx1, my1, mx2, my2 = moto
    pw, mw, mh = px2 - px1, mx2 - mx1, my2 - my1
    if pw <= 0 or mw <= 0 or mh <= 0:
        return False
    x_overlap = max(0.0, min(px2, mx2) - max(px1, mx1)) / min(pw, mw)
    top_above = py1 < my1 + 0.25 * mh            # head/torso above the bike
    bottom_in = py2 >= my1 + min_bottom_depth * mh   # legs reach well into the bike
    not_far_below = py2 < my2 + 0.5 * mh             # not someone standing in front of it
    tall_enough = (py2 - py1) >= min_height_ratio * mh
    return x_overlap >= min_x_overlap and top_above and bottom_in and not_far_below and tall_enough


def find_riders(person_boxes: np.ndarray, moto_boxes: np.ndarray, min_x_overlap: float = 0.5,
                min_bottom_depth: float = 0.5, min_height_ratio: float = 0.8) -> np.ndarray:
    """Indices of persons riding any motorcycle (several riders may share one bike)."""
    keep = [i for i, p in enumerate(person_boxes)
            if any(is_riding(p, m, min_x_overlap, min_bottom_depth, min_height_ratio) for m in moto_boxes)]
    return np.asarray(keep, np.int64)


def associate_heads(rider_boxes: np.ndarray, head_boxes: np.ndarray, min_containment: float = 0.5,
                    upper_fraction: float = 0.5) -> Dict[int, int]:
    """Assign each rider at most one head (and each head at most one rider).

    A head is a candidate for a rider when at least ``min_containment`` of the head box is
    inside the rider box and the head centre lies in the top ``upper_fraction`` of it.
    Pairs are assigned greedily: most-contained first, then the head closest to the top.
    """
    candidates = []
    for hi, h in enumerate(head_boxes):
        h_area = max((h[2] - h[0]) * (h[3] - h[1]), 1e-6)
        cy = (h[1] + h[3]) / 2
        for ri, r in enumerate(rider_boxes):
            iw = max(0.0, min(h[2], r[2]) - max(h[0], r[0]))
            ih = max(0.0, min(h[3], r[3]) - max(h[1], r[1]))
            containment = iw * ih / h_area
            rel_y = (cy - r[1]) / max(r[3] - r[1], 1e-6)
            if containment >= min_containment and rel_y <= upper_fraction:
                candidates.append((-containment, rel_y, hi, ri))
    candidates.sort()
    used_heads, assignment = set(), {}
    for _, _, hi, ri in candidates:
        if hi in used_heads or ri in assignment:
            continue
        assignment[ri] = hi
        used_heads.add(hi)
    return assignment
