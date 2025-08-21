#!/usr/bin/env bash
set -e
DEST=${1:-data/coco2017}
mkdir -p "$DEST"
cd "$DEST"
if [ ! -d "val2017" ]; then
  echo "Downloading val2017 images..."
  curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
  unzip -q val2017.zip && rm val2017.zip
fi
mkdir -p annotations
if [ ! -f "annotations/captions_val2017.json" ]; then
  echo "Downloading annotations..."
  curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations.zip
  unzip -q annotations.zip annotations/captions_val2017.json && rm annotations.zip
fi
echo "Done."
