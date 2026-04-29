#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-recordings/iphone/cat_camera_sample_20260429_115219.mp4}"
shift || true
exec .venv/bin/python tools/compare_tracking_modes.py "$VIDEO" \
  --left-config configs/iphone_fast20_bytetrack.yaml \
  --right-config configs/iphone_fast20_botsort.yaml \
  --left-title "Fast20 ByteTrack 416/conf0.07" \
  --right-title "Fast20 BoT-SORT 416/conf0.07" \
  "$@"
