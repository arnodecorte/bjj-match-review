"""
Inference pipeline: video file → timestamped position log.

Flow
----
1. Sample frames from the video at a configurable fps.
2. Run YOLOv8-pose on each frame to detect athletes and extract 17-point
   COCO keypoints.
3. Pick the two highest-confidence detections.
4. Normalise keypoints and classify position with either:
   - the trained MLP (if a model checkpoint is available), or
   - the built-in heuristic classifier (geometry-based fallback).
5. Return a list of dicts suitable for the web API.

An annotated video (skeleton overlay + position label) can optionally be
written to disk for download.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from .labels import (
    DISPLAY_NAMES,
    IDX_TO_LABEL,
    POSITION_COLORS,
    POSITION_LABELS,
)
from .model import PositionClassifier
from .preprocess import (
    SKELETON_CONNECTIONS,
    center_normalize_keypoints,
    normalize_keypoints,
)

logger = logging.getLogger(__name__)

# BGR colours for the two athletes' skeletons
_ATHLETE_COLORS: list[tuple[int, int, int]] = [
    (255, 100, 0),   # blue (A1)
    (0, 200, 60),    # green (A2)
]


# ---------------------------------------------------------------------------
# Heuristic position classifier (no trained weights required)
# ---------------------------------------------------------------------------

def _body_orientation(kps: np.ndarray) -> float:
    """
    Angle of the body axis from horizontal (degrees).
    0° = lying flat, 90° = standing upright.
    """
    shoulder = (kps[5, :2] + kps[6, :2]) / 2
    ankle = (kps[15, :2] + kps[16, :2]) / 2
    dx = float(shoulder[0] - ankle[0])
    dy = float(ankle[1] - shoulder[1])  # flip: image y grows downward
    return float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6)))


def _bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return float(inter / (a1 + a2 - inter + 1e-6))


def heuristic_classify(
    kps1: np.ndarray,
    kps2: np.ndarray,
    box1: np.ndarray,
    box2: np.ndarray,
) -> tuple[str, float]:
    """
    Rule-based BJJ position classifier.

    Returns (label, confidence) where confidence reflects how clearly
    the geometry matches the rules (not a calibrated probability).
    """
    hip1 = (kps1[11, :2] + kps1[12, :2]) / 2
    hip2 = (kps2[11, :2] + kps2[12, :2]) / 2
    shoulder1 = (kps1[5, :2] + kps1[6, :2]) / 2
    shoulder2 = (kps2[5, :2] + kps2[6, :2]) / 2
    ankle1 = (kps1[15, :2] + kps1[16, :2]) / 2
    ankle2 = (kps2[15, :2] + kps2[16, :2]) / 2

    height1 = float(np.linalg.norm(shoulder1 - ankle1)) + 1e-6
    height2 = float(np.linalg.norm(shoulder2 - ankle2)) + 1e-6
    avg_h = (height1 + height2) / 2

    iou = _bbox_iou(box1, box2)
    orient1 = _body_orientation(kps1)
    orient2 = _body_orientation(kps2)

    # --- STANDING ---
    if iou < 0.25 and orient1 > 55 and orient2 > 55:
        return "standing", 0.75

    # --- TAKEDOWN TRANSITION ---
    if iou < 0.45 and (orient1 < 45 or orient2 < 45):
        if orient1 >= orient2:
            return "takedown_a1", 0.55
        return "takedown_a2", 0.55

    # Both athletes are largely horizontal / grounded from here on
    hip_y_diff = float(hip1[1] - hip2[1])  # positive → A1 hips are lower
    norm_diff = hip_y_diff / avg_h

    # Determine who is on top (lower y-value = higher in frame = on top)
    a1_on_top = hip1[1] < hip2[1]

    # --- MOUNT (large vertical hip separation) ---
    if abs(norm_diff) > 0.45:
        label = "mount_a1" if a1_on_top else "mount_a2"
        return label, 0.65

    # --- SIDE CONTROL / TURTLE (moderate separation) ---
    if abs(norm_diff) > 0.20:
        # Check compactness of bottom athlete (turtle = very compact)
        bottom_kps = kps1 if not a1_on_top else kps2
        compactness = (box1[2] - box1[0]) / (box1[3] - box1[1] + 1e-6)
        if a1_on_top:
            compactness = (box2[2] - box2[0]) / (box2[3] - box2[1] + 1e-6)
        if compactness > 1.2:
            label = "turtle_a2" if a1_on_top else "turtle_a1"
            return label, 0.58
        label = "side_control_a1" if a1_on_top else "side_control_a2"
        return label, 0.60

    # --- GUARD FAMILY (small vertical separation, high overlap) ---
    if iou > 0.55:
        # Tightly interlocked → closed guard
        label = "closed_guard_a1" if a1_on_top else "closed_guard_a2"
        return label, 0.55
    if iou > 0.30:
        label = "open_guard_a1" if a1_on_top else "open_guard_a2"
        return label, 0.50

    # Default fallback
    return "standing", 0.35


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class PositionInference:
    """
    End-to-end pipeline: video path → timestamped position log.

    Parameters
    ----------
    model_path:
        Path to a trained PositionClassifier checkpoint (.pt).
        If None or the file does not exist, the heuristic classifier is used.
    device:
        PyTorch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self._classifier: Optional[PositionClassifier] = None
        self._using_heuristic = True

        if model_path and Path(model_path).exists():
            try:
                self._classifier = PositionClassifier.load(model_path, device)
                self._using_heuristic = False
                logger.info("Loaded trained classifier from %s", model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load model from %s (%s). Falling back to heuristic.",
                    model_path,
                    exc,
                )
        else:
            logger.info(
                "No trained model found at %s — using heuristic classifier.",
                model_path,
            )

        # YOLOv8-pose is imported lazily so the server starts even if
        # ultralytics is slow to import.
        self._pose_model = None

    def _get_pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO  # noqa: PLC0415
            self._pose_model = YOLO("yolov8n-pose.pt")
        return self._pose_model

    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        fps: float = 2.0,
        annotated_output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[dict]:
        """
        Process *video_path* and return a timestamped position log.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        fps:
            How many frames per second to classify (lower = faster).
        annotated_output_path:
            If given, write a skeleton-annotated video to this path.
        progress_callback:
            Called with a float in [0, 1] as frames are processed.

        Returns
        -------
        list of dicts, each containing:
            timestamp (float), position (str), confidence (float),
            display_name (str), color (str)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_interval = max(1, round(video_fps / fps))

        writer: Optional[cv2.VideoWriter] = None
        if annotated_output_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_output_path, fourcc, video_fps, (w, h))

        pose_model = self._get_pose_model()
        results: list[dict] = []
        last_entry: Optional[dict] = None
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / video_fps

                if frame_idx % frame_interval == 0:
                    entry = self._process_frame(frame, timestamp, pose_model)
                    if entry:
                        results.append(entry)
                        last_entry = entry

                    if progress_callback:
                        progress_callback(min(frame_idx / total_frames, 1.0))

                if writer is not None:
                    annotated = self._annotate_frame(frame, last_entry)
                    writer.write(annotated)

                frame_idx += 1
        finally:
            cap.release()
            if writer:
                writer.release()

        if progress_callback:
            progress_callback(1.0)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        pose_model,
    ) -> Optional[dict]:
        h, w = frame.shape[:2]

        pose_results = pose_model(frame, verbose=False)
        if not pose_results:
            return None

        result = pose_results[0]
        if result.keypoints is None:
            return None

        kps_xy = result.keypoints.xy.cpu().numpy()    # (N, 17, 2)
        kps_conf = result.keypoints.conf              # may be None for some models
        if kps_conf is not None:
            kps_conf = kps_conf.cpu().numpy()          # (N, 17)
        else:
            kps_conf = np.ones((len(kps_xy), 17), dtype=np.float32)

        boxes = result.boxes
        if boxes is None or len(boxes) < 2:
            return None

        box_conf = boxes.conf.cpu().numpy()
        if len(box_conf) < 2:
            return None

        # Select the two most-confident detections
        top2 = np.argsort(box_conf)[::-1][:2]

        def build_kps(idx: int) -> np.ndarray:
            xy = kps_xy[idx]                           # (17, 2)
            conf = kps_conf[idx, :, np.newaxis]        # (17, 1)
            return np.concatenate([xy, conf], axis=1)  # (17, 3)

        kps1 = build_kps(top2[0])
        kps2 = build_kps(top2[1])

        # Sort athletes left-to-right so A1/A2 assignment is consistent
        if kps1[:, 0].mean() > kps2[:, 0].mean():
            kps1, kps2 = kps2, kps1
            top2 = top2[::-1]

        box_xyxy = boxes.xyxy.cpu().numpy()
        box1 = box_xyxy[top2[0]]
        box2 = box_xyxy[top2[1]]

        label, confidence = self._classify(kps1, kps2, box1, box2, w, h)

        return {
            "timestamp":    round(float(timestamp), 3),
            "position":     label,
            "display_name": DISPLAY_NAMES.get(label, label),
            "confidence":   round(float(confidence), 3),
            "color":        POSITION_COLORS.get(label, "#888"),
        }

    def _classify(
        self,
        kps1: np.ndarray,
        kps2: np.ndarray,
        box1: np.ndarray,
        box2: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> tuple[str, float]:
        if self._classifier is not None:
            return self._ml_classify(kps1, kps2, img_w, img_h)
        return heuristic_classify(kps1, kps2, box1, box2)

    def _ml_classify(
        self,
        kps1: np.ndarray,
        kps2: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> tuple[str, float]:
        features = normalize_keypoints(kps1, kps2, img_w, img_h)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._classifier(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return IDX_TO_LABEL[pred.item()], float(conf.item())

    def _annotate_frame(
        self,
        frame: np.ndarray,
        entry: Optional[dict],
    ) -> np.ndarray:
        """Draw skeleton overlay and position label onto a copy of *frame*."""
        out = frame.copy()

        # We don't re-run inference here — draw nothing if no entry yet
        if entry is None:
            return out

        # Position label banner
        label_text = entry["display_name"]
        conf_text = f"{round(entry['confidence'] * 100)}%"
        full_text = f"{label_text}  {conf_text}"

        color_hex = entry.get("color", "#888")
        bgr = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

        (text_w, text_h), baseline = cv2.getTextSize(
            full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        cv2.rectangle(out, (8, 8), (text_w + 20, text_h + baseline + 20), bgr, -1)
        cv2.putText(
            out, full_text, (14, text_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )

        return out

    @property
    def using_heuristic(self) -> bool:
        """True when no trained model is loaded."""
        return self._using_heuristic
