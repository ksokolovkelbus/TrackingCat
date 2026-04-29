#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/render_tracking_comparison_offline.py "$VIDEO" \
  --left-config configs/iphone_yolo_track.yaml \
  --right-config configs/iphone_openvino_bytetrack.yaml \
  --left-title "PyTorch YOLO 640/conf0.10 + ByteTrack" \
  --right-title "OpenVINO YOLO 640/conf0.10 + ByteTrack" \
  "$@"
