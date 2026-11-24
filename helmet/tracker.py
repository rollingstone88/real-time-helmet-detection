"""Lightweight IoU tracker that gives riders stable IDs across video frames and smooths
their helmet status over recent frames (reduces flicker under occlusion / motion blur)."""
from __future__ import annotations

from collections import Counter, deque
from typing import List, Sequence, Tuple

import numpy as np

from .metrics import box_iou

UNKNOWN = "unknown"


class Track:
    def __init__(self, track_id: int, box: np.ndarray, history: int):
        self.id = track_id
        self.box = np.asarray(box, np.float64)
        self.statuses: deque = deque(maxlen=history)
        self.hits = 0
        self.missed = 0

    def smoothed_status(self) -> str:
        known = [s for s in self.statuses if s != UNKNOWN]
        if not known:
            return UNKNOWN
        counts = Counter(known)
        best = max(counts.values())
        for s in reversed(known):  # tie -> most recent observation wins
            if counts[s] == best:
                return s
        return UNKNOWN


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15, history: int = 15,
                 min_hits_for_violation: int = 3, violation_status: str = "no_helmet"):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.history = history
        self.min_hits_for_violation = min_hits_for_violation
        self.violation_status = violation_status
        self.tracks: List[Track] = []
        self.violators: set = set()
        self._next_id = 1

    def update(self, boxes: Sequence[Sequence[float]], statuses: Sequence[str]) -> List[Tuple[int, str]]:
        """Returns (track_id, smoothed_status) for every input box, in input order."""
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        ious = box_iou(boxes, np.array([t.box for t in self.tracks]).reshape(-1, 4))
        assigned = [-1] * len(boxes)
        used = set()
        if ious.size:
            for flat in np.argsort(-ious, axis=None):
                di, ti = np.unravel_index(flat, ious.shape)
                if ious[di, ti] < self.iou_threshold:
                    break
                if assigned[di] != -1 or ti in used:
                    continue
                assigned[di] = int(ti)
                used.add(int(ti))

        for ti, track in enumerate(self.tracks):
            if ti not in used:
                track.missed += 1
        out_tracks = []
        for di, ti in enumerate(assigned):
            if ti == -1:
                track = Track(self._next_id, boxes[di], self.history)
                self._next_id += 1
                self.tracks.append(track)
            else:
                track = self.tracks[ti]
            track.box = boxes[di]
            track.missed = 0
            track.hits += 1
            track.statuses.append(statuses[di])
            out_tracks.append(track)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        results = []
        for track in out_tracks:
            status = track.smoothed_status()
            if status == self.violation_status and track.hits >= self.min_hits_for_violation:
                self.violators.add(track.id)
            results.append((track.id, status))
        return results
