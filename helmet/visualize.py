"""OpenCV drawing: bounding boxes, class labels, confidence scores and a stats panel."""
from __future__ import annotations

import zlib
from typing import Optional, Tuple

import cv2
import numpy as np

from .pipeline import FrameResult

COLORS = {  # BGR
    "helmet": (60, 180, 60),
    "no_helmet": (40, 40, 230),
    "unknown": (160, 160, 160),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(name: str) -> Tuple[int, int, int]:
    if name in COLORS:
        return COLORS[name]
    rng = np.random.RandomState(zlib.crc32(name.encode()))  # stable across runs
    return tuple(int(c) for c in rng.randint(60, 230, 3))


def status_text(status: str) -> str:
    return status.replace("_", " ").upper()


def frame_scale(img: np.ndarray) -> float:
    """Text / line scale so labels stay readable on 300 px images and on 1080p video."""
    return float(np.clip(min(img.shape[:2]) / 600, 0.6, 1.6))


def draw_box(img: np.ndarray, box, color, text: str, scale: float = 1.0, label_at_bottom: bool = False) -> None:
    h, w = img.shape[:2]
    thickness, font_scale = max(1, int(round(2 * scale))), 0.5 * scale
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if not text:
        return
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, 1)
    if label_at_bottom:  # rider labels go under the box so they don't cover the head labels
        ty = y2 + th + 4 if y2 + th + base + 4 < h else y2 - base - 2
    else:  # above the box, or inside it when there is no room at the top
        ty = y1 - 4 if y1 - th - 6 >= 0 else y1 + th + 4
    tx = min(x1, max(0, w - tw - 4))
    cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 4, ty + base), color, -1)
    cv2.putText(img, text, (tx + 2, ty - 1), FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_results(img: np.ndarray, result: FrameResult, fps: Optional[float] = None,
                 violations: Optional[int] = None, show_riders: bool = True) -> np.ndarray:
    scale = frame_scale(img)
    for rider in result.riders if show_riders else []:
        tag = f"Rider #{rider.track_id}" if rider.track_id is not None else "Rider"
        draw_box(img, rider.box, color_for(rider.status), f"{tag}: {status_text(rider.status)}", scale,
                 label_at_bottom=True)
    for head in result.heads:
        draw_box(img, head.box, color_for(head.name), f"{head.name} {head.score:.2f}", scale)

    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    counts = {}
    for head in result.heads:
        counts[head.name] = counts.get(head.name, 0) + 1
    lines.append("Heads: " + ("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "0"))
    if show_riders and result.riders is not None:
        helmet = sum(r.status == "helmet" for r in result.riders)
        no_helmet = sum(r.status == "no_helmet" for r in result.riders)
        lines.append(f"Riders: {len(result.riders)}  helmet={helmet}  no_helmet={no_helmet}")
    if violations is not None:
        lines.append(f"Unique violators: {violations}")
    if result.enhanced:
        lines.append("Low-light enhancement ON")
    _draw_panel(img, lines, scale)
    return img


def _draw_panel(img: np.ndarray, lines, scale: float = 1.0) -> None:
    """Semi-transparent stats panel in the bottom-left corner (keeps the top free for labels)."""
    if not lines:
        return
    font_scale, pad, line_h = 0.45 * scale, int(6 * scale), int(22 * scale)
    width = min(img.shape[1], max(cv2.getTextSize(l, FONT, font_scale, 1)[0][0] for l in lines) + 2 * pad)
    height = min(img.shape[0], line_h * len(lines) + pad)
    y0 = img.shape[0] - height
    roi = img[y0:, :width]
    img[y0:, :width] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, y0 + pad + line_h * i + int(line_h * 0.55)), FONT, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
