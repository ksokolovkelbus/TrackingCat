# TrackingCatWIFI

Local cat detection/tracking for Ubuntu with YOLO26s + ByteTrack, zone overlays, alerts, recording, and pan/tilt aiming support.

The current base project preset is tuned for CPU use:

- inference size: `imgsz=384`
- detector confidence: `0.07`
- tracker: `ByteTrack`
- tracker config: `configs/bytetrack_cats.yaml`

The iPhone live mode is the high-quality preset:

- model source: `yolo26m.pt`
- runtime model: `yolo26m_384_openvino_model`
- `iou_threshold: 0.9`
- async YOLO inference enabled
- process every 2nd frame for lower latency

## Setup

```bash
cd ~/PycharmProjects/TrackingCatWIFI
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run all commands from the project directory.

## Main launch commands

### 1. Webcam safe aiming mode

Config: `configs/webcam_aiming_safe.yaml`

```bash
./run_camera.sh webcam-safe
```

Aliases:

```bash
./run_camera.sh webcam
./run_camera.sh aiming
```

### 2. Webcam visual tracking mode

Config: `configs/webcam_visual_tracking.yaml`

```bash
./run_camera.sh webcam-visual
```

Aliases:

```bash
./run_camera.sh visual
```

### 3. ESP32 Wi-Fi camera

Config: `configs/esp32_wifi.yaml`

```bash
./run_camera.sh esp32
```

Edit the ESP32 snapshot URL in:

```text
configs/esp32_wifi.yaml
source.stream_url
```

### 4. iPhone via IP Camera Lite

Config: `configs/iphone_ipcamera.yaml`

```bash
./run_camera.sh iphone
```

This is the high-quality live preset. It uses YOLO26m exported to OpenVINO at `imgsz=384` and async inference for smoother display.

If `yolo26m_384_openvino_model/` is missing, create it locally once:

```bash
.venv/bin/python - <<PY
from ultralytics import YOLO
YOLO('yolo26m.pt').export(format='openvino', imgsz=384, dynamic=False, half=False, int8=False)
PY
mv yolo26m_openvino_model yolo26m_384_openvino_model
```

Edit the iPhone stream URL in:

```text
configs/iphone_ipcamera.yaml
source.stream_url
```

### 5. iPad via IP Camera Lite

Config: `configs/ipad_ipcamera.yaml`

```bash
./run_camera.sh ipad
```

Edit the iPad stream URL in:

```text
configs/ipad_ipcamera.yaml
source.stream_url
```

### 6. iPad pan/tilt aiming

Config: `configs/ipad_pantilt.yaml`

```bash
./run_camera.sh pantilt
```

This mode is for manual aiming/servo calibration. It keeps the pan/tilt controls and laser toggling available.

## Test on a recorded video

Use the video-test wrapper. Default mode is realtime playback, capped to 30 FPS so heavy async models have time to return detections:

```bash
./run_test_video.sh realtime /path/to/video.mp4
```

Example:

```bash
./run_test_video.sh realtime recordings/cat_camera_sample_20260429_185638.mp4
```

Fast mode intentionally does not throttle playback and is useful only for stress/speed checks:

```bash
./run_test_video.sh fast recordings/cat_camera_sample_20260429_185638.mp4
```

The video path goes after the optional mode. You can pass normal app overrides after the path, for example:

```bash
./run_test_video.sh realtime recordings/cat.mp4 --conf-thres 0.09 --imgsz 384
```

## Record camera samples

Default recorder command:

```bash
./run_record_sample.sh
```

By default it records from `configs/webcam_aiming_safe.yaml` into `recordings/` and keeps recording until you stop it.

Record for a fixed number of seconds:

```bash
./run_record_sample.sh 30
```

Record from another camera config:

```bash
./run_record_sample.sh --config configs/iphone_ipcamera.yaml --output-dir recordings
```

## Zone editor

Use the same config names as the launch modes.

Webcam safe:

```bash
.venv/bin/python -m app.main --config configs/webcam_aiming_safe.yaml --zone-editor true --device cpu
```

Webcam visual:

```bash
.venv/bin/python -m app.main --config configs/webcam_visual_tracking.yaml --zone-editor true --device cpu
```

ESP32:

```bash
.venv/bin/python -m app.main --config configs/esp32_wifi.yaml --zone-editor true --device cpu
```

iPhone:

```bash
.venv/bin/python -m app.main --config configs/iphone_ipcamera.yaml --zone-editor true --device cpu
```

iPad:

```bash
.venv/bin/python -m app.main --config configs/ipad_ipcamera.yaml --zone-editor true --device cpu
```

Zone editor keys:

- left mouse: select zone or start rectangle
- right mouse: finish polygon
- `r`: rectangle mode
- `p`: polygon mode
- `f`: floor zone type
- `s`: surface zone type
- `x`: restricted zone type
- `n`: edit next zone name
- `u`: undo last point or remove selected/last zone
- `c`: clear current draft
- `d`: delete selected zone
- `w`: save zones to YAML
- `l`: reload zones from YAML
- `q`: quit editor

## Important files

```text
app/main.py                 main runtime
app/detector.py             YOLO detection / YOLO-native ByteTrack call
app/video_source.py         webcam/file/http/rtsp sources + latest-frame wrapper
app/overlay.py              visualization
app/zones.py                zone classification
app/surface_monitor.py      alert logic
app/pan_tilt.py             pan/tilt control
app/pan_tilt_auto.py        pan/tilt helpers

tools/record_camera_sample.py   camera sample recorder

configs/bytetrack_cats.yaml         final ByteTrack settings
configs/webcam_aiming_safe.yaml     webcam safe mode
configs/webcam_visual_tracking.yaml webcam visual mode
configs/esp32_wifi.yaml             ESP32 Wi-Fi camera
configs/iphone_ipcamera.yaml        iPhone IP Camera Lite
configs/ipad_ipcamera.yaml          iPad IP Camera Lite
configs/ipad_pantilt.yaml           iPad pan/tilt aiming
```

## Current ByteTrack settings

```yaml
tracker_type: bytetrack
track_high_thresh: 0.10
track_low_thresh: 0.03
new_track_thresh: 0.12
track_buffer: 120
match_thresh: 0.85
fuse_score: true
```

## Output and local files

Generated videos and local artifacts are ignored by git:

```text
recordings/
artifacts/
*_openvino_model/
```

Keep camera-specific URLs and zones in the YAML config files.
