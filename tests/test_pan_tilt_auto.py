import numpy as np

from app.models import PanTiltControlConfig
from app.pan_tilt_auto import LaserDotDetector, PanTiltCalibrationData, PanTiltCalibrationSample


def test_laser_dot_detector_finds_bright_red_dot() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[100:104, 150:154] = (0, 0, 255)
    detector = LaserDotDetector(PanTiltControlConfig(enabled=True))
    detection = detector.detect(frame)
    assert detection is not None
    assert abs(detection.center[0] - 152) <= 4
    assert abs(detection.center[1] - 102) <= 4


def test_calibration_data_predicts_angles_from_pixel() -> None:
    samples = [PanTiltCalibrationSample(pan_angle=40, tilt_angle=50, pixel_x=40, pixel_y=50, radius_px=2, area_px=6)]
    data = PanTiltCalibrationData(
        version=1,
        created_at=0.0,
        frame_width=200,
        frame_height=100,
        pan_coefficients=[100.0, 0.0, 0.0, 0.0, 0.0, 90.0],
        tilt_coefficients=[0.0, 50.0, 0.0, 0.0, 0.0, 45.0],
        fit_error_degrees=0.0,
        samples=samples,
    )
    pan, tilt = data.pixel_to_angles(100, 50)
    assert round(pan, 3) == 90.0
    assert round(tilt, 3) == 45.0


def test_calibrator_builds_local_targets() -> None:
    from app.pan_tilt_auto import PanTiltCalibrator
    from app.pan_tilt import PanTiltController
    import logging

    class DummyController(PanTiltController):
        def __init__(self):
            super().__init__(PanTiltControlConfig(enabled=True), logging.getLogger("test"))
        def _request_json(self, path, params=None):
            return {"pan_angle": 90, "tilt_angle": 90, "step_degrees": 3, "speed_mode": "medium", "laser_on": True, "connected": True}

    cfg = PanTiltControlConfig(enabled=True, auto_calibration_local_pan_step_degrees=8, auto_calibration_local_tilt_step_degrees=6, auto_calibration_local_pan_points=3, auto_calibration_local_tilt_points=3)
    cal = PanTiltCalibrator(cfg, DummyController(), logging.getLogger("test"))
    grid = cal._build_local_targets(90.0, 90.0)
    assert len(grid) == 9
    assert (90.0, 90.0) in grid
