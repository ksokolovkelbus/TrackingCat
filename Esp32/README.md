# ESP32 PanTilt firmware

Main firmware file used for the current PanTilt controller:
- `esp32_pantilt.ino`

Current purpose:
- Wi‑Fi control for pan/tilt + laser
- HTTP API for TrackingCat iPad PanTilt mode

Typical workflow:
1. Edit `esp32_pantilt.ino`
2. Compile/upload with `arduino-cli`
3. Test from TrackingCat `./run_camera.sh pantilt`

Current board target:
- ESP32-S3

Current known wiring defaults in firmware:
- Pan servo: GPIO17
- Tilt servo: GPIO18
- Laser: GPIO5
