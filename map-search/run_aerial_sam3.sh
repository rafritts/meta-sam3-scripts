#!/usr/bin/env bash
# Driver script for map-search/aerial_sam3.py
# Uncomment the example you want to run, or edit the active example at the bottom.
#
# Resolution guide — with --size 2048,2048:
#   --miles 5  --grid  5  →  25 tiles  ~0.7 ft/px
#   --miles 5  --grid  8  →  64 tiles  ~0.7 ft/px  (covers ~5x5 mi at street level)
#   --miles 5  --grid 10  → 100 tiles  ~0.7 ft/px
#   --miles 8  --grid 10  → 100 tiles  ~1.1 ft/px  (more area, slightly less zoom)
#   --miles 10 --grid 12  → 144 tiles  ~1.2 ft/px
#
# Color assignment is automatic and stable per run:
#   1st prompt → red, 2nd → blue, 3rd → green, 4th → orange, …

set -euo pipefail
cd "$(dirname "$0")/.."   # run from repo root so uv finds pyproject.toml

# ---------------------------------------------------------------------------
# 1. Ashburn, VA — 5-mile area, 10×10 grid (100 tiles at ~0.7 ft/px)
#    Full suburb sweep at street level.
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --address "Ashburn, VA" \
#   --miles 5 \
#   --grid 10 \
#   --prompts "house,swimming pool,car" \
#   --output-dir ./map-search/output/ashburn

# ---------------------------------------------------------------------------
# 2. Larger sweep — 8×8 grid over 8 miles (64 tiles, ~1.1 ft/px)
#    Good balance of coverage vs. tile count for a whole town.
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --address "Ashburn, VA" \
#   --miles 8 \
#   --grid 8 \
#   --prompts "house,swimming pool,car,solar panel" \
#   --output-dir ./map-search/output/ashburn_wide

# ---------------------------------------------------------------------------
# 3. Dense sweep — 12×12 over 6 miles (144 tiles, ~0.7 ft/px)
#    Maximum coverage at high resolution. Expect a long runtime.
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --address "Ashburn, VA" \
#   --miles 6 \
#   --grid 12 \
#   --prompts "house,swimming pool" \
#   --output-dir ./map-search/output/ashburn_dense

# ---------------------------------------------------------------------------
# 4. Different city — same pattern works anywhere in the US
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --address "Phoenix, AZ" \
#   --miles 8 \
#   --grid 10 \
#   --prompts "swimming pool,solar panel" \
#   --output-dir ./map-search/output/phoenix

# ---------------------------------------------------------------------------
# 5. By bounding box — explicit coords, explicit grid
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --bbox "-77.5200,39.0200,-77.4600,39.0700" \
#   --grid 8 \
#   --prompts "house,road,parking lot" \
#   --output-dir ./map-search/output/bbox_run

# ---------------------------------------------------------------------------
# 6. With GeoJSON export — for GIS / mapping downstream use
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --address "Ashburn, VA" \
#   --miles 5 \
#   --grid 10 \
#   --prompts "swimming pool" \
#   --geojson \
#   --output-dir ./map-search/output/ashburn_geojson

# ---------------------------------------------------------------------------
# 7. Loose threshold — maximize recall, filter false positives in post
# ---------------------------------------------------------------------------
# uv run map-search/aerial_sam3.py \
#   --address "Ashburn, VA" \
#   --miles 5 \
#   --grid 10 \
#   --prompts "solar panel" \
#   --threshold 0.15 \
#   --mask-cutoff 0.35 \
#   --output-dir ./map-search/output/ashburn_solar

# ---------------------------------------------------------------------------
# Active example — edit freely
# ---------------------------------------------------------------------------
uv run map-search/aerial_sam3.py \
  --address "Ashburn, VA" \
  --miles 5 \
  --grid 10 \
  --prompts "house,swimming pool,car" \
  --batch-size 4 \
  --workers 16 \
  --output-dir ./map-search/output/ashburn
