#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true

exec .venv/bin/python tools/benchmark_tracking_modes.py "$VIDEO" \
  --configs \
    configs/iphone_yolo_track.yaml \
    configs/iphone_openvino_bytetrack.yaml \
    configs/iphone_pytorch_balanced_bytetrack.yaml \
    configs/iphone_openvino_balanced_bytetrack.yaml \
  --names \
    "PyTorch 640/conf0.10" \
    "OpenVINO 640/conf0.10" \
    "PyTorch 800/conf0.07" \
    "OpenVINO 800/conf0.07" \
  "$@"
