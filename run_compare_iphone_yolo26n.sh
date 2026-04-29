#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/compare_tracking_modes.py "$VIDEO" \
  --left-config configs/iphone_yolo26n_fast_openvino.yaml \
  --right-config configs/iphone_yolo26n_quality.yaml \
  --left-title "YOLO26n Fast OpenVINO 384/conf0.05" \
  --right-title "YOLO26n Quality 640/conf0.05" \
  "$@"
