#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-recordings/cat_camera_sample_20260429_185638.mp4}"
shift || true
exec .venv/bin/python tools/render_model_conf_row.py "$VIDEO" \
  --model yolo26s.pt \
  --label YOLO26s \
  --imgsz 384 \
  --confs 0.05 0.07 0.09 \
  "$@"
