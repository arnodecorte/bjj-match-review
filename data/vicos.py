"""
ViCoS Lab BJJ dataset loader.

Reference:
  Hudovernik & Skočaj (2022) — "Video-Based Detection of Combat Positions
  and Automatic Scoring in Jiu-jitsu" (ACM MMSports'22)

Dataset URL:  https://vicos.si/resources/jiujitsu/
Images:       http://data.vicos.si/datasets/JuiJuitsu/images.zip
Annotations:  http://data.vicos.si/datasets/JuiJuitsu/annotations.json

After downloading, run this module directly to verify the dataset:

    python -m data.vicos --data-dir data/vicos --verify
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_IMAGES = 120_279
EXPECTED_CLASSES = 18


# ---------------------------------------------------------------------------
# Dataset inspection helpers
# ---------------------------------------------------------------------------

def load_annotations(data_dir: Path) -> dict:
    """Load and return the raw COCO-format annotations dict."""
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotations not found at {ann_path}.\n"
            "Run  bash scripts/download_vicos.sh  to download the dataset."
        )
    with open(ann_path) as fh:
        return json.load(fh)


def summarise(data_dir: Path) -> dict:
    """Return a summary dict with counts and class distribution."""
    coco = load_annotations(data_dir)

    n_images = len(coco.get("images", []))
    n_annotations = len(coco.get("annotations", []))
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    by_category: dict[int, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        by_category[ann.get("category_id", -1)] += 1

    return {
        "n_images": n_images,
        "n_annotations": n_annotations,
        "categories": categories,
        "per_category": dict(by_category),
    }


def verify(data_dir: Path) -> bool:
    """Check that the dataset looks complete."""
    ok = True
    summary = summarise(data_dir)

    logger.info("Images in annotations: %d (expected ≈%d)", summary["n_images"], EXPECTED_IMAGES)
    if summary["n_images"] < EXPECTED_IMAGES * 0.95:
        logger.warning("Image count is lower than expected — dataset may be incomplete.")
        ok = False

    logger.info("Categories: %d (expected %d)", len(summary["categories"]), EXPECTED_CLASSES)
    for cid, name in sorted(summary["categories"].items()):
        count = summary["per_category"].get(cid, 0)
        logger.info("  [%2d] %-30s  %d annotations", cid, name, count)

    images_dir = data_dir / "images"
    if not images_dir.exists():
        logger.warning("images/ directory not found at %s", images_dir)
        ok = False
    else:
        n_files = sum(1 for _ in images_dir.glob("*.jpg"))
        logger.info("Image files on disk: %d", n_files)

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ViCoS dataset utilities")
    parser.add_argument("--data-dir", type=Path, default=Path("data/vicos"))
    parser.add_argument("--verify",   action="store_true",
                        help="Verify the dataset is complete")
    args = parser.parse_args()

    if args.verify:
        ok = verify(args.data_dir)
        raise SystemExit(0 if ok else 1)
    else:
        summary = summarise(args.data_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
