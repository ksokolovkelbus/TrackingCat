#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/render_model_resolution_row.py "$VIDEO" \
  --model yolo26m.pt \
  --label YOLO26m \
  --conf 0.05 \
  --resolutions 640 384 256 \
  "$@"
