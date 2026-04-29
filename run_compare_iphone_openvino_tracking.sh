#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/compare_tracking_modes.py "$VIDEO" \
  --left-config configs/iphone_openvino_bytetrack.yaml \
  --right-config configs/iphone_openvino_botsort.yaml \
  --left-title "iPhone OpenVINO + ByteTrack" \
  --right-title "iPhone OpenVINO + BoT-SORT" \
  "$@"
