from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.models import PanTiltControlConfig
from app.pan_tilt import PanTiltController


@dataclass(slots=True)
class LaserDotDetection:
    center: tuple[int, int]
    radius_px: float
    area_px: float
    score: float


@dataclass(slots=True)
class PanTiltCalibrationSample:
    pan_angle: float
    tilt_angle: float
    pixel_x: float
    pixel_y: float
    radius_px: float
    area_px: float


@dataclass(slots=True)
class PanTiltCalibrationData:
    version: int
    created_at: float
    frame_width: int
    frame_height: int
    pan_coefficients: list[float]
    tilt_coefficients: list[float]
    fit_error_degrees: float
    samples: list[PanTiltCalibrationSample]

    def pixel_to_angles(self, x: float, y: float) -> tuple[float, float]:
        features = _feature_vector(x, y, self.frame_width, self.frame_height)
        pan = float(features @ np.array(self.pan_coefficients, dtype=np.float64))
        tilt = float(features @ np.array(self.tilt_coefficients, dtype=np.float64))
        return pan, tilt


class LaserDotDetector:
    def __init__(self, config: PanTiltControlConfig) -> None:
        self._config = config

    def detect(self, frame: np.ndarray | None) -> LaserDotDetection | None:
        if frame is None or frame.size == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, self._config.laser_saturation_min, self._config.laser_value_min], dtype=np.uint8)
        upper1 = np.array([12, 255, 255], dtype=np.uint8)
        lower2 = np.array([168, self._config.laser_saturation_min, self._config.laser_value_min], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        b_channel, g_channel, r_channel = cv2.split(frame)
        red_dominance = cv2.subtract(r_channel, cv2.max(b_channel, g_channel))
        red_mask = cv2.inRange(r_channel, self._config.laser_red_min, 255)
        dominance_mask = cv2.inRange(red_dominance, self._config.laser_red_delta, 255)
        mask = cv2.bitwise_and(hsv_mask, cv2.bitwise_and(red_mask, dominance_mask))
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 32, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: LaserDotDetection | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._config.laser_min_area_px or area > self._config.laser_max_area_px:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius <= 0.0:
                continue
            circle_area = np.pi * radius * radius
            circularity = 0.0 if circle_area <= 0.0 else min(1.0, area / circle_area)
            if circularity < 0.15:
                continue
            score = area * (0.6 + circularity)
            candidate = LaserDotDetection(
                center=(int(round(cx)), int(round(cy))),
                radius_px=float(radius),
                area_px=area,
                score=score,
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best


class PanTiltCalibrator:
    def __init__(
        self,
        config: PanTiltControlConfig,
        controller: PanTiltController,
        logger: logging.Logger,
        detector: LaserDotDetector | None = None,
    ) -> None:
        self._config = config
        self._controller = controller
        self._logger = logger
        self._detector = detector or LaserDotDetector(config)
        self._calibration: PanTiltCalibrationData | None = None

    @property
    def detector(self) -> LaserDotDetector:
        return self._detector

    @property
    def calibration(self) -> PanTiltCalibrationData | None:
        return self._calibration

    def load(self) -> PanTiltCalibrationData | None:
        path = Path(self._config.calibration_artifact_path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._calibration = PanTiltCalibrationData(
            version=int(payload.get("version", 1)),
            created_at=float(payload.get("created_at", time.time())),
            frame_width=int(payload["frame_width"]),
            frame_height=int(payload["frame_height"]),
            pan_coefficients=[float(v) for v in payload["pan_coefficients"]],
            tilt_coefficients=[float(v) for v in payload["tilt_coefficients"]],
            fit_error_degrees=float(payload.get("fit_error_degrees", 0.0)),
            samples=[PanTiltCalibrationSample(**sample) for sample in payload.get("samples", [])],
        )
        return self._calibration

    def calibrate(self, fetch_frame: Callable[[], np.ndarray | None]) -> PanTiltCalibrationData:
        self._controller.maybe_refresh_state(force=True)
        state = self._controller.state
        original_pan = state.pan_angle
        original_tilt = state.tilt_angle
        original_speed = state.speed_mode
        laser_was_on = state.laser_on
        samples: list[PanTiltCalibrationSample] = []
        last_frame: np.ndarray | None = None
        try:
            if not laser_was_on:
                self._controller.set_laser(True)
                time.sleep(0.2)
            if original_speed != "slow":
                self._controller.set_speed_mode("slow")
            for index, (pan_angle, tilt_angle) in enumerate(self._build_grid(), start=1):
                self._logger.info("Auto calibration sample %d -> pan=%s tilt=%s", index, pan_angle, tilt_angle)
                self._controller.target_angles(int(round(pan_angle)), int(round(tilt_angle)))
                time.sleep(self._config.auto_calibration_settle_seconds)
                detection, frame = self._wait_for_detection(fetch_frame)
                if detection is None or frame is None:
                    self._logger.warning("Laser dot not found at pan=%s tilt=%s", pan_angle, tilt_angle)
                    continue
                last_frame = frame
                samples.append(
                    PanTiltCalibrationSample(
                        pan_angle=float(pan_angle),
                        tilt_angle=float(tilt_angle),
                        pixel_x=float(detection.center[0]),
                        pixel_y=float(detection.center[1]),
                        radius_px=float(detection.radius_px),
                        area_px=float(detection.area_px),
                    )
                )
            minimum = max(6, self._config.auto_calibration_min_samples)
            if len(samples) < minimum:
                raise RuntimeError(f"Calibration collected only {len(samples)} usable samples; need at least {minimum}.")
            if last_frame is None:
                raise RuntimeError("Calibration finished without any valid frame.")
            data = self._fit(samples=samples, frame_width=int(last_frame.shape[1]), frame_height=int(last_frame.shape[0]))
            self._calibration = data
            self._save(data)
            return data
        finally:
            try:
                self._controller.target_angles(int(round(original_pan)), int(round(original_tilt)))
            except Exception:
                self._logger.warning("Failed to restore original pan/tilt after calibration.", exc_info=True)
            try:
                if original_speed != self._controller.state.speed_mode:
                    self._controller.set_speed_mode(original_speed)
            except Exception:
                self._logger.warning("Failed to restore original speed mode after calibration.", exc_info=True)
            try:
                if not laser_was_on and self._controller.state.laser_on:
                    self._controller.set_laser(False)
            except Exception:
                self._logger.warning("Failed to restore laser state after calibration.", exc_info=True)

    def aim_at_pixel(self, x: float, y: float) -> tuple[float, float]:
        calibration = self._calibration or self.load()
        if calibration is None:
            raise RuntimeError("PanTilt calibration is not loaded.")
        pan_angle, tilt_angle = calibration.pixel_to_angles(x, y)
        self._controller.target_angles(int(round(pan_angle)), int(round(tilt_angle)))
        return pan_angle, tilt_angle

    def _wait_for_detection(self, fetch_frame: Callable[[], np.ndarray | None]) -> tuple[LaserDotDetection | None, np.ndarray | None]:
        deadline = time.monotonic() + self._config.auto_calibration_detection_timeout_seconds
        best_detection: LaserDotDetection | None = None
        best_frame: np.ndarray | None = None
        while time.monotonic() < deadline:
            frame = fetch_frame()
            if frame is None or frame.size == 0:
                time.sleep(0.02)
                continue
            detection = self._detector.detect(frame)
            if detection is not None and (best_detection is None or detection.score > best_detection.score):
                best_detection = detection
                best_frame = frame.copy()
                if detection.score >= max(8.0, self._config.laser_min_area_px * 2.0):
                    break
        return best_detection, best_frame

    def _build_grid(self) -> list[tuple[float, float]]:
        pans = np.linspace(
            self._config.auto_calibration_pan_min_angle,
            self._config.auto_calibration_pan_max_angle,
            num=max(2, self._config.auto_calibration_grid_cols),
            dtype=np.float64,
        )
        tilts = np.linspace(
            self._config.auto_calibration_tilt_min_angle,
            self._config.auto_calibration_tilt_max_angle,
            num=max(2, self._config.auto_calibration_grid_rows),
            dtype=np.float64,
        )
        grid: list[tuple[float, float]] = []
        for row_index, tilt in enumerate(tilts):
            pan_iter = pans if row_index % 2 == 0 else pans[::-1]
            for pan in pan_iter:
                grid.append((float(pan), float(tilt)))
        return grid

    def _fit(self, samples: list[PanTiltCalibrationSample], frame_width: int, frame_height: int) -> PanTiltCalibrationData:
        design = np.vstack([_feature_vector(sample.pixel_x, sample.pixel_y, frame_width, frame_height) for sample in samples])
        pan_targets = np.array([sample.pan_angle for sample in samples], dtype=np.float64)
        tilt_targets = np.array([sample.tilt_angle for sample in samples], dtype=np.float64)
        pan_coefficients, *_ = np.linalg.lstsq(design, pan_targets, rcond=None)
        tilt_coefficients, *_ = np.linalg.lstsq(design, tilt_targets, rcond=None)
        pan_predictions = design @ pan_coefficients
        tilt_predictions = design @ tilt_coefficients
        fit_error = np.sqrt(np.mean(((pan_predictions - pan_targets) ** 2 + (tilt_predictions - tilt_targets) ** 2) / 2.0))
        return PanTiltCalibrationData(
            version=1,
            created_at=time.time(),
            frame_width=frame_width,
            frame_height=frame_height,
            pan_coefficients=[float(value) for value in pan_coefficients.tolist()],
            tilt_coefficients=[float(value) for value in tilt_coefficients.tolist()],
            fit_error_degrees=float(fit_error),
            samples=samples,
        )

    def _save(self, data: PanTiltCalibrationData) -> None:
        path = Path(self._config.calibration_artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": data.version,
            "created_at": data.created_at,
            "frame_width": data.frame_width,
            "frame_height": data.frame_height,
            "pan_coefficients": data.pan_coefficients,
            "tilt_coefficients": data.tilt_coefficients,
            "fit_error_degrees": data.fit_error_degrees,
            "samples": [asdict(sample) for sample in data.samples],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _feature_vector(x: float, y: float, frame_width: int, frame_height: int) -> np.ndarray:
    width = max(1.0, float(frame_width))
    height = max(1.0, float(frame_height))
    nx = (float(x) / width) - 0.5
    ny = (float(y) / height) - 0.5
    return np.array([nx, ny, nx * ny, nx * nx, ny * ny, 1.0], dtype=np.float64)
