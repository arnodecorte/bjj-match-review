"""Focus-lock utilities for robust active fighter pair selection.

Ranks candidate pose detections using:
- detector confidence
- center-frame bias
- temporal continuity
- engagement/proximity
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np


def bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return float(inter / (a1 + a2 - inter + 1e-6))


def _center_score(box: np.ndarray, frame_w: int, frame_h: int) -> float:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    dx = abs(cx - frame_w / 2) / (frame_w / 2 + 1e-6)
    dy = abs(cy - frame_h / 2) / (frame_h / 2 + 1e-6)
    return max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5))


def _engagement_score(box1: np.ndarray, box2: np.ndarray, frame_w: int, frame_h: int) -> float:
    c1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    c2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    dist = np.linalg.norm(c1 - c2)
    max_dist = (frame_w**2 + frame_h**2) ** 0.5
    proximity = 1.0 - min(dist / (max_dist + 1e-6), 1.0)
    overlap = bbox_iou(box1, box2)
    return 0.7 * proximity + 0.3 * overlap


def _temporal_score(candidate_pair: tuple[np.ndarray, np.ndarray], previous_pair: Optional[tuple[np.ndarray, np.ndarray]]) -> float:
    if previous_pair is None:
        return 0.5
    c1, c2 = candidate_pair
    p1, p2 = previous_pair
    direct = (bbox_iou(c1, p1) + bbox_iou(c2, p2)) / 2
    swapped = (bbox_iou(c1, p2) + bbox_iou(c2, p1)) / 2
    return max(direct, swapped)


def select_active_pair(boxes_xyxy: np.ndarray, confidences: np.ndarray, frame_w: int, frame_h: int, previous_pair: Optional[tuple[np.ndarray, np.ndarray]] = None) -> Optional[tuple[int, int]]:
    if len(boxes_xyxy) < 2:
        return None

    best_score = -1.0
    best_pair = None

    for i, j in combinations(range(len(boxes_xyxy)), 2):
        b1, b2 = boxes_xyxy[i], boxes_xyxy[j]
        confidence_score = float((confidences[i] + confidences[j]) / 2)
        center_score = (_center_score(b1, frame_w, frame_h) + _center_score(b2, frame_w, frame_h)) / 2
        temporal_score = _temporal_score((b1, b2), previous_pair)
        engagement_score = _engagement_score(b1, b2, frame_w, frame_h)

        score = (
            0.45 * confidence_score
            + 0.25 * center_score
            + 0.20 * temporal_score
            + 0.10 * engagement_score
        )

        if score > best_score:
            best_score = score
            best_pair = (i, j)

    return best_pair
