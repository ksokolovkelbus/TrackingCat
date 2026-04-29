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
  webcam-manycats|manycats|multi-cats|multi-cat)
    CONFIG="configs/webcam_manycats.yaml"
    ;;
  webcam-yolo-track|yolo-track|native-track)
    CONFIG="configs/webcam_yolo_track.yaml"
    ;;
  webcam-botsort-track|botsort-track)
    CONFIG="configs/webcam_botsort_track.yaml"
    ;;
  esp32|wifi|esp32_wifi)
    CONFIG="configs/esp32_wifi.yaml"
    ;;
  iphone-ipcam|iphone|ipcam)
    CONFIG="configs/iphone_ipcamera.yaml"
    ;;
  iphone-yolo-track|iphone-bytetrack)
    CONFIG="configs/iphone_yolo_track.yaml"
    ;;
  iphone-botsort-track|iphone-botsort)
    CONFIG="configs/iphone_botsort_track.yaml"
    ;;
  iphone-motion-bytetrack|iphone-yolo-motion|iphone-motion-yolo)
    CONFIG="configs/iphone_motion_bytetrack.yaml"
    ;;
  iphone-motion-botsort)
    CONFIG="configs/iphone_motion_botsort.yaml"
    ;;
  iphone-openvino-bytetrack|iphone-ov-bytetrack)
    CONFIG="configs/iphone_openvino_bytetrack.yaml"
    ;;
  iphone-openvino-botsort|iphone-ov-botsort)
    CONFIG="configs/iphone_openvino_botsort.yaml"
    ;;
  iphone-openvino-balanced-bytetrack|iphone-ov-balanced-bytetrack)
    CONFIG="configs/iphone_openvino_balanced_bytetrack.yaml"
    ;;
  iphone-openvino-balanced-botsort|iphone-ov-balanced-botsort)
    CONFIG="configs/iphone_openvino_balanced_botsort.yaml"
    ;;
  iphone-fast20-bytetrack|iphone-fast20|iphone-fast)
    CONFIG="configs/iphone_fast20_bytetrack.yaml"
    ;;
  iphone-fast20-botsort)
    CONFIG="configs/iphone_fast20_botsort.yaml"
    ;;
  iphone-fast20-openvino|iphone-fast20-ov)
    CONFIG="configs/iphone_fast20_openvino_bytetrack.yaml"
    ;;
  ipad-ipcam|ipad|ipadcamera)
    CONFIG="configs/ipad_ipcamera.yaml"
    ;;
  ipad-manycats|ipad-multi-cats|ipad-multi-cat)
    CONFIG="configs/ipad_manycats.yaml"
    ;;
  ipad-yolo-track|ipad-native-track)
    CONFIG="configs/ipad_yolo_track.yaml"
    ;;
  ipad-botsort-track|ipad-botsort)
    CONFIG="configs/ipad_botsort_track.yaml"
    ;;
  ipad-pantilt|pantilt|ipad-pan-tilt)
    CONFIG="configs/ipad_pantilt.yaml"
    ;;
  *)
    echo "Usage: ./run_camera.sh [webcam-safe|webcam-visual|webcam-manycats|webcam-yolo-track|webcam-botsort-track|esp32|iphone|iphone-yolo-track|iphone-botsort-track|iphone-motion-bytetrack|iphone-motion-botsort|iphone-openvino-bytetrack|iphone-openvino-botsort|iphone-openvino-balanced-bytetrack|iphone-openvino-balanced-botsort|iphone-fast20-bytetrack|iphone-fast20-botsort|iphone-fast20-openvino|ipad|ipad-manycats|ipad-yolo-track|ipad-botsort-track|pantilt] [extra app args...]" >&2
    exit 2
    ;;
esac

exec .venv/bin/python -m app.main --config "$CONFIG" --device cpu "$@"
