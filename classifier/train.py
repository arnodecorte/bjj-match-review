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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .labels import LABEL_TO_IDX, NUM_CLASSES, POSITION_LABELS
from .model import PositionClassifier
from .preprocess import normalize_keypoints

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vicos_annotations(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse ViCoS annotations.json and return (X, y) arrays.

    X shape: (N, 102)  — normalised COCO keypoints for both athletes
    y shape: (N,)      — integer class labels (0–17)

    The ViCoS annotations use COCO format where each image has exactly two
    person annotations (one per athlete).  The image-level position label
    is encoded as category_id (1-indexed in the JSON, 0-indexed in y).
    """
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"{ann_path} not found. Run 'bash scripts/download_vicos.sh' first."
        )

    with open(ann_path) as fh:
        coco = json.load(fh)

    # Build a mapping from image_id → list of annotation dicts
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    # Build category name → class index mapping
    # ViCoS categories are 1-indexed; map to our 0-indexed labels.
    cat_id_to_label: dict[int, str] = {}
    for cat in coco.get("categories", []):
        # Normalise the category name to our label format
        name = cat["name"].lower().replace(" ", "_").replace("-", "_")
        if name in LABEL_TO_IDX:
            cat_id_to_label[cat["id"]] = name
        else:
            # Attempt a prefix match
            for lbl in POSITION_LABELS:
                if lbl.startswith(name) or name.startswith(lbl.split("_")[0]):
                    cat_id_to_label[cat["id"]] = lbl
                    break

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    skipped = 0

    for image_id, anns in by_image.items():
        if len(anns) < 2:
            skipped += 1
            continue

        # Take the first two annotations (each athlete)
        a1, a2 = anns[0], anns[1]

        # COCO keypoints are stored flat: [x1, y1, v1, x2, y2, v2, ...]
        def parse_kps(ann: dict) -> np.ndarray:
            flat = ann["keypoints"]  # 51 values
            kps = np.array(flat, dtype=np.float32).reshape(17, 3)
            return kps

        kps1 = parse_kps(a1)
        kps2 = parse_kps(a2)

        # Image dimensions for normalisation
        img = next(
            (i for i in coco["images"] if i["id"] == image_id), None
        )
        if img is None:
            skipped += 1
            continue

        img_w, img_h = img.get("width", 1920), img.get("height", 1080)
        features = normalize_keypoints(kps1, kps2, img_w, img_h)

        # Category (position label) — use the first annotation's category
        cat_id = a1.get("category_id", a2.get("category_id", -1))
        label_name = cat_id_to_label.get(cat_id)
        if label_name is None:
            # Fall back to 0-indexed direct mapping
            label_idx = cat_id - 1
            if 0 <= label_idx < NUM_CLASSES:
                label_name = POSITION_LABELS[label_idx]
            else:
                skipped += 1
                continue

        X_list.append(features)
        y_list.append(LABEL_TO_IDX[label_name])

    if skipped:
        logger.warning("Skipped %d images (missing annotations).", skipped)
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
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    X, y = load_vicos_annotations(data_dir)

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

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    model = PositionClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
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
        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_acc=%.4f", epoch, epochs, avg_loss, val_acc
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(
                str(output_path),
                extra={"epoch": epoch, "val_acc": val_acc, "labels": POSITION_LABELS},
            )
            logger.info("  ↳ Saved best model (val_acc=%.4f)", val_acc)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("Model saved to %s", output_path)


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
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
