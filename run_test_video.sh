#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VIDEO="${1:-}"
if [[ -z "$VIDEO" || "$VIDEO" == "--help" || "$VIDEO" == "-h" ]]; then
  echo "Usage: ./run_test_video.sh /path/to/video.mp4 [extra app args...]" >&2
  echo "Example: ./run_test_video.sh recordings/cat_camera_sample_20260429_185638.mp4" >&2
  exit 2
fi
shift || true

exec .venv/bin/python -m app.main \
  --config configs/iphone_ipcamera.yaml \
  --source file \
  --input "$VIDEO" \
  --device cpu \
  "$@"
