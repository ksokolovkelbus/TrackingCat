#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/benchmark_tracking_modes.py "$VIDEO" \
  --configs \
    configs/iphone_fast20_bytetrack.yaml \
    configs/iphone_fast20_openvino_bytetrack.yaml \
    configs/iphone_fast20_botsort.yaml \
    configs/iphone_yolo_track.yaml \
    configs/iphone_openvino_bytetrack.yaml \
  --names \
    "Fast20 ByteTrack 416/conf0.07" \
    "Fast20 OpenVINO ByteTrack 416/conf0.07" \
    "Fast20 BoT-SORT 416/conf0.07" \
    "Old PyTorch 640/conf0.10" \
    "Old OpenVINO 640/conf0.10" \
  "$@"
