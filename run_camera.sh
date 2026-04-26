#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-webcam-safe}"
shift || true

case "$MODE" in
  webcam|webcam-safe|aiming|aiming-safe)
    CONFIG="configs/webcam_aiming_safe.yaml"
    ;;
  webcam-visual|visual|tracking-visual)
    CONFIG="configs/webcam_visual_tracking.yaml"
    ;;
  esp32|wifi|esp32_wifi)
    CONFIG="configs/esp32_wifi.yaml"
    ;;
  iphone-ipcam|iphone|ipcam)
    CONFIG="configs/iphone_ipcamera.yaml"
    ;;
  ipad-ipcam|ipad|ipadcamera)
    CONFIG="configs/ipad_ipcamera.yaml"
    ;;
  *)
    echo "Usage: ./run_camera.sh [webcam-safe|webcam-visual|esp32|iphone|ipad] [extra app args...]" >&2
    exit 2
    ;;
esac

exec .venv/bin/python -m app.main --config "$CONFIG" --device cpu "$@"
