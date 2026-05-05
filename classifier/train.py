"""
Training script for the BJJ position classifier.

Usage (after downloading the ViCoS dataset):

    python -m classifier.train \\
        --data-dir data/vicos \\
        --epochs 50 \\
        --batch-size 256 \\
        --lr 1e-3 \\
        --output models/classifier.pt

The ViCoS dataset must be downloaded first:

    bash scripts/download_vicos.sh

Expected directory layout::

    data/vicos/
        annotations.json     # COCO-format annotations
        images/              # JPEG frames

See DATA_SOURCES.md for the full dataset specification.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from .labels import LABEL_TO_IDX, NUM_CLASSES, POSITION_LABELS
from .model import PositionClassifier
from .preprocess import normalize_keypoints

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_LR_KP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


def augment_feature_batch(
    x: torch.Tensor,
    noise_std: float,
    conf_dropout: float,
    athlete_swap_prob: float,
    hflip_prob: float,
    zoom_jitter: float,
) -> torch.Tensor:
    """
    Apply keypoint-level augmentations that mimic real video variance.

    Input shape: (B, 102) with normalised [x, y, conf] for 2 athletes.
    """
    if x.ndim != 2 or x.shape[1] != 102:
        return x

    out = x.clone()
    kps = out.view(-1, 2, 17, 3)

    # Camera shake / detector noise.
    if noise_std > 0:
        kps[:, :, :, :2] += torch.randn_like(kps[:, :, :, :2]) * noise_std

    # Mild zoom / distance variation around frame centre.
    if zoom_jitter > 0:
        scale = 1.0 + (torch.rand((kps.shape[0], 1, 1, 1), device=x.device) * 2 - 1) * zoom_jitter
        kps[:, :, :, :2] = (kps[:, :, :, :2] - 0.5) * scale + 0.5

    # Occlusion simulation by reducing keypoint confidence.
    if conf_dropout > 0:
        drop = (torch.rand_like(kps[:, :, :, 2]) < conf_dropout).float()
        kps[:, :, :, 2] *= (1.0 - drop)

    # Left/right camera mirroring with left/right landmark remap.
    if hflip_prob > 0:
        flip_mask = torch.rand((kps.shape[0],), device=x.device) < hflip_prob
        if flip_mask.any():
            kps[flip_mask, :, :, 0] = 1.0 - kps[flip_mask, :, :, 0]
            for li, ri in _LR_KP_PAIRS:
                tmp = kps[flip_mask, :, li, :].clone()
                kps[flip_mask, :, li, :] = kps[flip_mask, :, ri, :]
                kps[flip_mask, :, ri, :] = tmp

    # Swap athlete order to reduce overfitting to left/right assignment.
    if athlete_swap_prob > 0:
        swap_mask = torch.rand((kps.shape[0],), device=x.device) < athlete_swap_prob
        if swap_mask.any():
            tmp = kps[swap_mask, 0, :, :].clone()
            kps[swap_mask, 0, :, :] = kps[swap_mask, 1, :, :]
            kps[swap_mask, 1, :, :] = tmp

    kps[:, :, :, :2] = torch.clamp(kps[:, :, :, :2], 0.0, 1.0)
    kps[:, :, :, 2] = torch.clamp(kps[:, :, :, 2], 0.0, 1.0)
    return kps.view(-1, 102)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vicos_annotations(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse ViCoS annotations.json and return (X, y) arrays.

    X shape: (N, 102)  — normalised COCO keypoints for both athletes
    y shape: (N,)      — integer class labels (0–17)

    The actual ViCoS annotations.json is a flat list of dicts, each with:
      - 'position': string label (e.g. 'standing', 'mount1', '5050_guard')
      - 'pose1': list of 17 keypoints [[x, y, conf], ...] for athlete 1
      - 'pose2': list of 17 keypoints [[x, y, conf], ...] for athlete 2
      - 'image': image filename
      - 'frame': frame index
    """
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"{ann_path} not found. Run 'bash scripts/download_vicos.sh' first."
        )

    with open(ann_path) as fh:
        records = json.load(fh)

    # ViCoS position name → our POSITION_LABELS index
    # Suffix '1'/'2' maps to '_a1'/'_a2'; '5050_guard' maps to '50_50'.
    _VICOS_MAP: dict[str, str] = {
        "standing":       "standing",
        "takedown1":      "takedown_a1",
        "takedown2":      "takedown_a2",
        "open_guard1":    "open_guard_a1",
        "open_guard2":    "open_guard_a2",
        "half_guard1":    "half_guard_a1",
        "half_guard2":    "half_guard_a2",
        "closed_guard1":  "closed_guard_a1",
        "closed_guard2":  "closed_guard_a2",
        "5050_guard":     "50_50",
        "side_control1":  "side_control_a1",
        "side_control2":  "side_control_a2",
        "mount1":         "mount_a1",
        "mount2":         "mount_a2",
        "back1":          "back_a1",
        "back2":          "back_a2",
        "turtle1":        "turtle_a1",
        "turtle2":        "turtle_a2",
    }

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    skipped = 0

    for rec in records:
        label_name = _VICOS_MAP.get(rec.get("position", ""))
        if label_name is None:
            skipped += 1
            continue

        pose1 = rec.get("pose1")
        pose2 = rec.get("pose2")
        if pose1 is None and pose2 is None:
            skipped += 1
            continue

        _zeros = [[0.0, 0.0, 0.0]] * 17
        if pose1 is None or len(pose1) != 17:
            pose1 = _zeros
        if pose2 is None or len(pose2) != 17:
            pose2 = _zeros

        kps1 = np.array(pose1, dtype=np.float32)  # (17, 3)
        kps2 = np.array(pose2, dtype=np.float32)  # (17, 3)

        # Keypoints are already in pixel space; use a nominal 1920×1080 frame
        # for normalisation (ViCoS uses consistent resolution).
        features = normalize_keypoints(kps1, kps2, 1920, 1080)

        X_list.append(features)
        y_list.append(LABEL_TO_IDX[label_name])

    if skipped:
        logger.warning("Skipped %d records (unknown label or missing pose).", skipped)
    logger.info("Loaded %d samples, %d classes.", len(X_list), NUM_CLASSES)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: Path,
    output_path: Path,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device_str: str = "cpu",
    max_samples: int | None = None,
    patience: int = 8,
    min_delta: float = 1e-4,
    label_smoothing: float = 0.05,
    augment: bool = True,
    aug_noise_std: float = 0.01,
    aug_conf_dropout: float = 0.06,
    aug_athlete_swap_prob: float = 0.25,
    aug_hflip_prob: float = 0.25,
    aug_zoom_jitter: float = 0.08,
) -> None:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Training on device: %s", device)

    X, y = load_vicos_annotations(data_dir)
    if max_samples is not None and 0 < max_samples < len(X):
        logger.info("Subsampling dataset to %d samples (stratified).", max_samples)
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=max_samples,
            stratify=y,
            random_state=42,
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )
    logger.info(
        "Train: %d  |  Val: %d", len(X_train), len(X_val)
    )

    def make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    logger.info("Train class counts: %s", class_counts.tolist())

    inv_class_freq = 1.0 / np.maximum(class_counts, 1)
    sample_weights = inv_class_freq[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = make_loader(X_train, y_train, shuffle=False)
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = make_loader(X_val, y_val, shuffle=False)

    model = PositionClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = torch.tensor(inv_class_freq, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if augment:
                X_batch = augment_feature_batch(
                    X_batch,
                    noise_std=aug_noise_std,
                    conf_dropout=aug_conf_dropout,
                    athlete_swap_prob=aug_athlete_swap_prob,
                    hflip_prob=aug_hflip_prob,
                    zoom_jitter=aug_zoom_jitter,
                )
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(X_batch)
        scheduler.step()

        # --- validate ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        val_acc = correct / total
        avg_loss = train_loss / len(X_train)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "val_acc": float(val_acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_acc=%.4f", epoch, epochs, avg_loss, val_acc
        )

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            stale_epochs = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(
                str(output_path),
                extra={"epoch": epoch, "val_acc": val_acc, "labels": POSITION_LABELS},
            )
            logger.info("  ↳ Saved best model (val_acc=%.4f)", val_acc)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val_acc=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_acc,
                )
                break

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "duration_sec": round(time.time() - started, 2),
                "history": history,
            },
            fh,
            indent=2,
        )

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("Model saved to %s", output_path)
    logger.info("Metrics saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BJJ position classifier")
    parser.add_argument("--data-dir",    type=Path, default=Path("data/vicos"))
    parser.add_argument("--output",      type=Path, default=Path("models/classifier.pt"))
    parser.add_argument("--epochs",      type=int,  default=50)
    parser.add_argument("--batch-size",  type=int,  default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val-split",   type=float, default=0.15)
    parser.add_argument("--device",      type=str,  default="cpu",
                        help="PyTorch device string, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional stratified cap on total samples for quick experiments")
    parser.add_argument("--patience",    type=int, default=8,
                        help="Early-stopping patience on validation accuracy")
    parser.add_argument("--min-delta",   type=float, default=1e-4,
                        help="Minimum validation-accuracy improvement to reset patience")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Cross-entropy label smoothing factor")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable keypoint feature augmentations")
    parser.add_argument("--aug-noise-std", type=float, default=0.01)
    parser.add_argument("--aug-conf-dropout", type=float, default=0.06)
    parser.add_argument("--aug-athlete-swap-prob", type=float, default=0.25)
    parser.add_argument("--aug-hflip-prob", type=float, default=0.25)
    parser.add_argument("--aug-zoom-jitter", type=float, default=0.08)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device_str=args.device,
        max_samples=args.max_samples,
        patience=args.patience,
        min_delta=args.min_delta,
        label_smoothing=args.label_smoothing,
        augment=not args.no_augment,
        aug_noise_std=args.aug_noise_std,
        aug_conf_dropout=args.aug_conf_dropout,
        aug_athlete_swap_prob=args.aug_athlete_swap_prob,
        aug_hflip_prob=args.aug_hflip_prob,
        aug_zoom_jitter=args.aug_zoom_jitter,
    )


if __name__ == "__main__":
    main()
