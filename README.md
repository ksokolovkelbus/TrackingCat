# TrackingCatWIFI

Local cat detection/tracking for Ubuntu with YOLO26s + ByteTrack, zone overlays, alerts, recording, and pan/tilt aiming support.

The current project preset is tuned for CPU use:

- model: `YOLO26s`
- inference size: `imgsz=384`
- detector confidence: `0.07`
- tracker: `ByteTrack`
- tracker config: `configs/bytetrack_cats.yaml`

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

Use the single video-test wrapper:

```bash
./run_test_video.sh /path/to/video.mp4
```

Example:

```bash
./run_test_video.sh recordings/cat_camera_sample_20260429_185638.mp4
```

The video path goes as the **first argument** after `run_test_video.sh`.

You can pass normal app overrides after the path, for example:

```bash
./run_test_video.sh recordings/cat.mp4 --conf-thres 0.09 --imgsz 384
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
