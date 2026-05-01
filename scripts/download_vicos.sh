#!/usr/bin/env bash
# Download the ViCoS Lab BJJ dataset (images + annotations).
#
# Dataset:  https://vicos.si/resources/jiujitsu/
# License:  CC BY-NC-SA 4.0  (non-commercial use only)
#
# Usage:    bash scripts/download_vicos.sh [DATA_DIR]
#
# The default DATA_DIR is  data/vicos/  relative to the repo root.

set -euo pipefail

DATA_DIR="${1:-data/vicos}"
IMAGES_URL="http://data.vicos.si/datasets/JuiJuitsu/images.zip"
ANNOTATIONS_URL="http://data.vicos.si/datasets/JuiJuitsu/annotations.json"

mkdir -p "$DATA_DIR"

echo "==> Downloading annotations…"
if [ ! -f "$DATA_DIR/annotations.json" ]; then
  curl -L --progress-bar "$ANNOTATIONS_URL" -o "$DATA_DIR/annotations.json"
  echo "    Saved to $DATA_DIR/annotations.json"
else
  echo "    Already exists, skipping."
fi

echo "==> Downloading images (≈ several GB, may take a while)…"
if [ ! -f "$DATA_DIR/images.zip" ]; then
  curl -L --progress-bar "$IMAGES_URL" -o "$DATA_DIR/images.zip"
  echo "    Saved to $DATA_DIR/images.zip"
else
  echo "    Already exists, skipping."
fi

echo "==> Extracting images…"
if [ ! -d "$DATA_DIR/images" ]; then
  unzip -q "$DATA_DIR/images.zip" -d "$DATA_DIR"
  echo "    Extracted to $DATA_DIR/images/"
else
  echo "    images/ directory already exists, skipping extraction."
fi

echo ""
echo "✅  ViCoS dataset ready in $DATA_DIR/"
echo ""
echo "   Verify with:  python -m data.vicos --data-dir $DATA_DIR --verify"
echo "   Train with:   make train"
