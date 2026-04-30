#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="realtime"
if [[ "${1:-}" == "realtime" || "${1:-}" == "fast" ]]; then
  MODE="$1"
  shift
fi

VIDEO="${1:-}"
if [[ -z "$VIDEO" || "$VIDEO" == "--help" || "$VIDEO" == "-h" ]]; then
  echo "Usage: ./run_test_video.sh [realtime|fast] /path/to/video.mp4 [extra app args...]" >&2
  echo "Examples:" >&2
  echo "  ./run_test_video.sh realtime recordings/cat_camera_sample_20260429_185638.mp4" >&2
  echo "  ./run_test_video.sh fast recordings/cat_camera_sample_20260429_185638.mp4" >&2
  echo "" >&2
  echo "Default mode is realtime. Use fast only when you want maximum processing/display speed." >&2
  exit 2
fi
shift || true

REALTIME_ARGS=(--playback-realtime true --playback-fps 30)
if [[ "$MODE" == "fast" ]]; then
  REALTIME_ARGS=(--playback-realtime false)
fi

exec .venv/bin/python -m app.main \
  --config configs/iphone_ipcamera.yaml \
  --source file \
  --input "$VIDEO" \
  --device cpu \
  "${REALTIME_ARGS[@]}" \
  "$@"
