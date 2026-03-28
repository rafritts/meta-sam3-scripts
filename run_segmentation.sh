#!/usr/bin/env bash
# Driver script for demo_segmentation.py
# Uncomment the example you want to run.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Download an image and segment by text prompt
#    The image is saved to images/ so subsequent runs skip the download.
# ---------------------------------------------------------------------------
# uv run demo_segmentation.py \
#   --download "http://images.cocodataset.org/val2017/000000077595.jpg" \
#   --text "cat" \
#   --output segmentation_result.png

# ---------------------------------------------------------------------------
# 2. Use whatever image is already in images/ (no download needed)
# ---------------------------------------------------------------------------
# uv run demo_segmentation.py \
#   --text "dog"

# ---------------------------------------------------------------------------
# 3. Point at a specific local image
# ---------------------------------------------------------------------------
# uv run demo_segmentation.py \
#   --image images/my_photo.jpg \
#   --text "person"

# ---------------------------------------------------------------------------
# 4. Text prompt + positive bounding box to focus the search
# ---------------------------------------------------------------------------
# uv run demo_segmentation.py \
#   --image images/my_photo.jpg \
#   --text "ear" \
#   --box 100 150 500 450

# ---------------------------------------------------------------------------
# 5. Text prompt + negative bounding box to exclude a region
# ---------------------------------------------------------------------------
# uv run demo_segmentation.py \
#   --image images/my_photo.jpg \
#   --text "handle" \
#   --neg-box 40 183 318 204

# ---------------------------------------------------------------------------
# 6. Tune detection sensitivity
#    --detection-threshold : filters out entire detections below this score
#    --mask-cutoff         : lower = bigger/looser mask, higher = tighter mask
# ---------------------------------------------------------------------------
# uv run demo_segmentation.py \
#   --image images/my_photo.jpg \
#   --text "chair" \
#   --detection-threshold 0.3 \
#   --mask-cutoff 0.4 \
#   --output chair_result.png

# ---------------------------------------------------------------------------
# Active example — edit freely
# ---------------------------------------------------------------------------
uv run demo_segmentation.py \
  --image images/map.png \
  --text "field" \
  --detection-threshold 0.1 \
  --mask-cutoff 0.5
  #--top 4 \