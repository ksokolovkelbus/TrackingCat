# TrackingCat

TrackingCat is a local cat detection and tracking system with zone-based alerts.

It runs on Ubuntu and supports three camera sources:
- built-in or USB webcam
- ESP32 Wi-Fi snapshot camera
- iPhone running IP Camera Lite (MJPEG over HTTP)

The project is tuned for CPU inference and is intended for real-time household monitoring, not cloud deployment.

## Features

- cat detection with YOLO models
- single or multi-cat tracking
- zone editor with per-camera saved layouts
- restricted / floor zone logic
- alert sound playback
- alert video recording with metadata
- support for multiple camera configs
- CPU-friendly presets

## Camera modes

### 1. Webcam safe aiming mode
Config: `configs/webcam_aiming_safe.yaml`

Use when you want the main webcam profile with practical defaults for aiming and alert zones.

Run:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
./run_camera.sh webcam-safe
```

### 2. Webcam visual tracking mode
Config: `configs/webcam_visual_tracking.yaml`

Use when you want richer visual tracking feedback.

Run:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
./run_camera.sh webcam-visual
```

### 3. ESP32 Wi-Fi camera
Config: `configs/esp32_wifi.yaml`

This mode reads JPEG snapshots from the ESP32 UVC Wi-Fi camera.

Expected endpoint:
- `http://192.168.8.140/snapshot.jpg`

Run:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
./run_camera.sh esp32
```

### 4. iPhone via IP Camera Lite
Config: `configs/iphone_ipcamera.yaml`

This mode reads the live MJPEG stream from IP Camera Lite.

Current expected endpoint:
- `http://192.168.8.181:8081/video`

Run:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
./run_camera.sh iphone
```

If the iPhone IP changes, update `source.stream_url` in `configs/iphone_ipcamera.yaml`.

## Zone editor

Each camera mode keeps its own config, so zones are independent between webcam, ESP32, and iPhone.

Examples:

### Webcam safe
```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m app.main --config configs/webcam_aiming_safe.yaml --zone-editor true --device cpu
```

### Webcam visual
```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m app.main --config configs/webcam_visual_tracking.yaml --zone-editor true --device cpu
```

### ESP32
```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m app.main --config configs/esp32_wifi.yaml --zone-editor true --device cpu
```

### iPhone
```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m app.main --config configs/iphone_ipcamera.yaml --zone-editor true --device cpu
```

## Configuration overview

Main sections in each YAML config:

- `source` , camera source and reconnect behavior
- `detector` , model and confidence thresholds
- `overlay` , crosshair, labels, FPS, counters
- `tracking` , multi-cat and track retention behavior
- `scene_zones` , floor / restricted zones
- `surface_alert` , alert trigger logic
- `alert_recording` , saved alert clips and metadata
- `resize` , optional post-capture resize

## Recommended current usage

### Webcam
- good default for local testing
- best when you want stable interactive tuning

### ESP32
- useful when you need a lightweight remote camera
- current camera is low resolution, so expect lower accuracy than webcam or iPhone

### iPhone
- best image quality of the current setups
- currently preferred with wide camera, 4K, 30 FPS at source
- actual TrackingCat processing FPS may be much lower, which is normal on CPU

## Installation

Create and use a Python virtual environment, then install dependencies:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Tests

Run tests with:

```bash
cd ~/PycharmProjects/TrackingCatWIFI
.venv/bin/python -m pytest -q
```

## Project structure

```text
app/        application code
configs/    camera-specific YAML configs
tests/      test suite
sounds/     alert sounds
run_camera.sh
README.md
```

## Notes

- This repository intentionally does not include local virtual environments, logs, alert recordings, backups, certificates, or model weights.
- Large model files such as `*.pt` are excluded from git.
- The project is designed for local use on the iMac Ubuntu machine.
