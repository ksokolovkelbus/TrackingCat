#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec .venv/bin/python tools/record_camera_sample.py --help
fi

EXTRA_ARGS=()
if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  EXTRA_ARGS+=(--seconds "$1")
  shift
fi

exec .venv/bin/python tools/record_camera_sample.py \
  --config configs/iphone_yolo_track.yaml \
  --output-dir recordings/iphone \
  --show \
  "${EXTRA_ARGS[@]}" \
  "$@"
