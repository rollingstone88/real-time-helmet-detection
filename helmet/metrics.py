"""Object-detection metrics: precision, recall, F1, mAP and a confusion matrix.

Matching follows the COCO protocol: within each image and class, predictions are visited
in descending score order and each one is matched to the *unmatched* ground-truth box with
the highest IoU >= threshold. AP uses COCO 101-point interpolated precision.

* mAP@0.5        AP at IoU 0.50, averaged over classes
* mAP@0.5:0.95   AP averaged over IoU 0.50, 0.55, ..., 0.95 and over classes
* precision / recall / F1  at IoU 0.50 for detections with score >= conf_threshold

Labels are 1..C (0 is background), matching the detector output.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_predictions(pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                      iou_thresholds: Sequence[float]) -> np.ndarray:
    """pred_boxes must be sorted by descending score. Returns tp flags (T, N)."""
    n, g = len(pred_boxes), len(gt_boxes)
    tp = np.zeros((len(iou_thresholds), n), bool)
    if n == 0 or g == 0:
        return tp
    ious = box_iou(pred_boxes, gt_boxes)
    for ti, thr in enumerate(iou_thresholds):
        matched = np.zeros(g, bool)
        for i in range(n):
            cand = np.flatnonzero(~matched & (ious[i] >= thr))
            if cand.size:
                j = cand[np.argmax(ious[i, cand])]
                matched[j] = True
                tp[ti, i] = True
    return tp


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO-style 101-point interpolated AP. NaN when the class has no ground truth."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order].astype(np.float64)
    tpc = np.cumsum(tp)
    fpc = np.cumsum(1.0 - tp)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    q = np.zeros(len(RECALL_POINTS))
    valid = idx < len(recall)
    q[valid] = precision[idx[valid]]
    return float(q.mean())


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


class DetectionEvaluator:
    def __init__(self, class_names: Sequence[str], max_detections: int = 100):
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.max_detections = max_detections
        self.reset()

    def reset(self) -> None:
        C = self.num_classes
        self._scores: List[List[np.ndarray]] = [[] for _ in range(C)]
        self._tp: List[List[np.ndarray]] = [[] for _ in range(C)]
        self._n_gt = np.zeros(C, np.int64)
        self._images: List[tuple] = []  # kept for the confusion matrix

    @property
    def num_images(self) -> int:
        return len(self._images)

    def add(self, pred: Dict[str, np.ndarray], gt: Dict[str, np.ndarray]) -> None:
        """pred: boxes (N,4), scores (N,), labels (N,); gt: boxes (M,4), labels (M,)."""
        pb = np.asarray(pred["boxes"], np.float64).reshape(-1, 4)
        ps = np.asarray(pred["scores"], np.float64).reshape(-1)
        pl = np.asarray(pred["labels"], np.int64).reshape(-1)
        gb = np.asarray(gt["boxes"], np.float64).reshape(-1, 4)
        gl = np.asarray(gt["labels"], np.int64).reshape(-1)
        for c in range(1, self.num_classes + 1):
            m = pl == c
            boxes, scores = pb[m], ps[m]
            order = np.argsort(-scores, kind="mergesort")[: self.max_detections]
            gt_c = gb[gl == c]
            self._tp[c - 1].append(match_predictions(boxes[order], gt_c, IOU_THRESHOLDS))
            self._scores[c - 1].append(scores[order])
            self._n_gt[c - 1] += len(gt_c)
        self._images.append((pb, ps, pl, gb, gl))

    def _class_arrays(self, ci: int):
        if self._scores[ci]:
            return np.concatenate(self._scores[ci]), np.concatenate(self._tp[ci], axis=1)
        return np.zeros(0), np.zeros((len(IOU_THRESHOLDS), 0), bool)

    def compute(self, conf_threshold: float = 0.5) -> dict:
        per_class = {}
        ap = np.full((self.num_classes, len(IOU_THRESHOLDS)), np.nan)
        totals = np.zeros(3, np.int64)  # tp, fp, fn
        for ci, name in enumerate(self.class_names):
            scores, tp = self._class_arrays(ci)
            n_gt = int(self._n_gt[ci])
            for ti in range(len(IOU_THRESHOLDS)):
                ap[ci, ti] = average_precision(tp[ti], scores, n_gt)
            keep = scores >= conf_threshold
            c_tp = int(tp[0][keep].sum())
            c_fp = int(keep.sum()) - c_tp
            c_fn = n_gt - c_tp
            totals += (c_tp, c_fp, c_fn)
            p, r, f1 = _prf(c_tp, c_fp, c_fn)
            per_class[name] = {
                "ap50": ap[ci, 0], "ap75": ap[ci, 5], "ap50_95": float(np.mean(ap[ci])),
                "precision": p, "recall": r, "f1": f1,
                "tp": c_tp, "fp": c_fp, "fn": c_fn, "num_gt": n_gt,
            }
        p, r, f1 = _prf(*totals)
        valid = ~np.isnan(ap[:, 0])
        macro = [per_class[n] for n, v in zip(self.class_names, valid) if v]
        best_t, best_f1 = self.best_f1_threshold()
        return {
            "num_images": self.num_images,
            "conf_threshold": conf_threshold,
            "map50": float(np.mean(ap[valid, 0])) if valid.any() else float("nan"),
            "map75": float(np.mean(ap[valid, 5])) if valid.any() else float("nan"),
            "map50_95": float(np.mean(ap[valid])) if valid.any() else float("nan"),
            "precision": p, "recall": r, "f1": f1,  # micro-averaged over classes
            "macro_precision": float(np.mean([m["precision"] for m in macro])) if macro else 0.0,
            "macro_recall": float(np.mean([m["recall"] for m in macro])) if macro else 0.0,
            "macro_f1": float(np.mean([m["f1"] for m in macro])) if macro else 0.0,
            "best_f1_conf_threshold": best_t, "best_f1": best_f1,
            "per_class": per_class,
            "confusion_matrix": self.confusion_matrix(conf_threshold).tolist(),
            "confusion_labels": self.class_names + ["background"],
        }

    # ------------------------------------------------------------------ curves
    def pr_curves(self) -> Dict[str, dict]:
        """Precision-recall curve per class at IoU 0.5 (for plotting)."""
        curves = {}
        for ci, name in enumerate(self.class_names):
            scores, tp = self._class_arrays(ci)
            n_gt = int(self._n_gt[ci])
            if n_gt == 0 or len(scores) == 0:
                curves[name] = {"recall": np.zeros(0), "precision": np.zeros(0), "ap50": average_precision(tp[0], scores, n_gt)}
                continue
            order = np.argsort(-scores, kind="mergesort")
            t = tp[0][order].astype(np.float64)
            tpc, fpc = np.cumsum(t), np.cumsum(1 - t)
            curves[name] = {"recall": tpc / n_gt, "precision": tpc / (tpc + fpc),
                            "ap50": average_precision(tp[0], scores, n_gt)}
        return curves

    def f1_curve(self, thresholds: Optional[np.ndarray] = None):
        """Micro-averaged precision / recall / F1 (IoU 0.5) for each confidence threshold."""
        thresholds = np.linspace(0, 1, 101) if thresholds is None else np.asarray(thresholds)
        arrays = [self._class_arrays(ci) for ci in range(self.num_classes)]
        n_gt = int(self._n_gt.sum())
        P, R, F = [], [], []
        for t in thresholds:
            tp = sum(int(a[1][0][a[0] >= t].sum()) for a in arrays)
            n_pred = sum(int((a[0] >= t).sum()) for a in arrays)
            p, r, f1 = _prf(tp, n_pred - tp, n_gt - tp)
            P.append(p); R.append(r); F.append(f1)
        return thresholds, np.array(P), np.array(R), np.array(F)

    def best_f1_threshold(self):
        t, _, _, f = self.f1_curve()
        if len(f) == 0 or f.max() == 0:
            return 0.5, 0.0
        i = int(np.argmax(f))
        return float(t[i]), float(f[i])

    # ------------------------------------------------------------------ confusion
    def confusion_matrix(self, conf_threshold: float = 0.5, iou_threshold: float = 0.5) -> np.ndarray:
        """Rows = ground truth, columns = prediction; last row/col = background.

        Class-agnostic matching shows helmet <-> no_helmet confusions explicitly:
        (gt background, pred c) = false positive, (gt c, pred background) = missed object.
        """
        C = self.num_classes
        cm = np.zeros((C + 1, C + 1), np.int64)
        for pb, ps, pl, gb, gl in self._images:
            keep = ps >= conf_threshold
            pb, ps, pl = pb[keep], ps[keep], pl[keep]
            order = np.argsort(-ps, kind="mergesort")
            pb, pl = pb[order], pl[order]
            ious = box_iou(pb, gb)
            matched = np.zeros(len(gb), bool)
            for i in range(len(pb)):
                cand = np.flatnonzero(~matched & (ious[i] >= iou_threshold)) if len(gb) else []
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
                    matched[j] = True
                    cm[gl[j] - 1, pl[i] - 1] += 1
                else:
                    cm[C, pl[i] - 1] += 1
            for j in np.flatnonzero(~matched):
                cm[gl[j] - 1, C] += 1
        return cm


def format_metrics_table(metrics: dict) -> str:
    lines = [f"{'class':<14}{'AP50':>8}{'AP50-95':>9}{'P':>8}{'R':>8}{'F1':>8}{'GT':>7}"]
    for name, m in metrics["per_class"].items():
        lines.append(f"{name:<14}{_fmt(m['ap50']):>8}{_fmt(m['ap50_95']):>9}{m['precision']:>8.3f}"
                     f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['num_gt']:>7d}")
    lines.append(f"{'all':<14}{_fmt(metrics['map50']):>8}{_fmt(metrics['map50_95']):>9}"
                 f"{metrics['precision']:>8.3f}{metrics['recall']:>8.3f}{metrics['f1']:>8.3f}"
                 f"{sum(m['num_gt'] for m in metrics['per_class'].values()):>7d}")
    return "\n".join(lines)


def _fmt(v) -> str:
    return "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
