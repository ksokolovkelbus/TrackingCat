#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/benchmark_tracking_modes.py "$VIDEO" \
  --configs \
    configs/iphone_yolo26n_maxfps.yaml \
    configs/iphone_yolo26n_fast_openvino.yaml \
    configs/iphone_yolo26n_quality.yaml \
    configs/iphone_fast20_openvino_bytetrack.yaml \
  --names \
    "YOLO26n maxfps 256/conf0.05" \
    "YOLO26n fast OpenVINO 384/conf0.05" \
    "YOLO26n quality 640/conf0.05" \
    "YOLO26s fast20 OpenVINO 416/conf0.07" \
  "$@"
