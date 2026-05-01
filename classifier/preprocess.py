"""
Keypoint normalisation utilities.

All functions accept two (17, 3) numpy arrays of COCO keypoints
  kps[:, 0] = x (pixel)
  kps[:, 1] = y (pixel)
  kps[:, 2] = confidence [0, 1]
and return a flat feature vector of shape (102,) ready for the MLP.
"""

from __future__ import annotations

import numpy as np

# COCO 17-point landmark names (index = position in keypoint array)
COCO_KEYPOINTS: list[str] = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

NUM_KP = 17

# Skeleton connectivity for drawing
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]


def _mid(kps: np.ndarray, *indices: int) -> np.ndarray:
    """Return the xy midpoint of the specified keypoint indices."""
    return kps[list(indices), :2].mean(axis=0)


def normalize_keypoints(
    kps1: np.ndarray,
    kps2: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Simple pixel-space normalisation to [0, 1].

    Fastest option; works well when the camera is relatively static
    (typical of ViCoS training data).
    """
    k1 = kps1.astype(np.float32).copy()
    k2 = kps2.astype(np.float32).copy()

    k1[:, 0] /= img_width
    k1[:, 1] /= img_height
    k2[:, 0] /= img_width
    k2[:, 1] /= img_height

    k1[:, 2] = np.clip(k1[:, 2], 0.0, 1.0)
    k2[:, 2] = np.clip(k2[:, 2], 0.0, 1.0)

    return np.concatenate([k1.flatten(), k2.flatten()])  # (102,)


def center_normalize_keypoints(
    kps1: np.ndarray,
    kps2: np.ndarray,
) -> np.ndarray:
    """
    Body-centre + scale normalisation — more robust to varying camera
    distance and position.  Uses the combined shoulder/hip cluster as
    the reference scale.
    """
    k1 = kps1.astype(np.float32).copy()
    k2 = kps2.astype(np.float32).copy()

    # Anchor: centroid of all xy
    all_xy = np.concatenate([k1[:, :2], k2[:, :2]], axis=0)
    centre = all_xy.mean(axis=0)

    # Scale: std of shoulder + hip points
    ref = np.concatenate([
        k1[5:7, :2], k1[11:13, :2],
        k2[5:7, :2], k2[11:13, :2],
    ])
    scale = float(np.std(ref)) + 1e-6

    k1[:, :2] = (k1[:, :2] - centre) / scale
    k2[:, :2] = (k2[:, :2] - centre) / scale

    return np.concatenate([k1.flatten(), k2.flatten()])  # (102,)
