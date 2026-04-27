from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
    circularity: float
    mean_red: float
    bbox: tuple[int, int, int, int]


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


@dataclass(slots=True)
class CalibrationProgress:
    running: bool = False
    sample_index: int = 0
    total_samples: int = 0
    current_target: tuple[float, float] | None = None
    stage: str = "idle"
    started_at: float = 0.0
    confirmed_frames: int = 0


class LaserDotDetector:
    def __init__(self, config: PanTiltControlConfig) -> None:
        self._config = config

    def detect(self, frame: np.ndarray | None, *, expected_center: tuple[int, int] | None = None, reference: LaserDotDetection | None = None, max_distance_px: float | None = None) -> LaserDotDetection | None:
        candidates = self.detect_candidates(frame)
        if not candidates:
            return None
        best: LaserDotDetection | None = None
        best_score = float('-inf')
        for candidate in candidates:
            score = candidate.score
            if expected_center is not None:
                dist = _distance(candidate.center, expected_center)
                if max_distance_px is not None and dist > max_distance_px:
                    continue
                score -= dist * 0.03
            if reference is not None:
                radius_delta = abs(candidate.radius_px - reference.radius_px) / max(reference.radius_px, 1.0)
                area_delta = abs(candidate.area_px - reference.area_px) / max(reference.area_px, 1.0)
                if radius_delta > self._config.auto_calibration_reference_radius_tolerance:
                    score -= 120.0 * radius_delta
                if area_delta > self._config.auto_calibration_reference_area_tolerance:
                    score -= 90.0 * area_delta
            if best is None or score > best_score:
                best = candidate
                best_score = score
        return best

    def detect_candidates(self, frame: np.ndarray | None) -> list[LaserDotDetection]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        b_channel, g_channel, r_channel = cv2.split(frame)
        dominance = cv2.subtract(r_channel, cv2.max(b_channel, g_channel))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        base_mask = (
            (r_channel >= self._config.laser_red_min)
            & (dominance >= self._config.laser_red_delta)
            & (sat >= self._config.laser_saturation_min)
            & (val >= self._config.laser_value_min)
        )
        if not np.any(base_mask):
            return []

        seed_mask = base_mask & (r_channel >= self._config.laser_peak_min_value)
        if not np.any(seed_mask):
            return []
        seed_response = (dominance.astype(np.float32) * 1.4) + (r_channel.astype(np.float32) * 2.2)
        seed_response[~seed_mask] = 0.0
        blurred = cv2.GaussianBlur(seed_response, (0, 0), 1.2)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        if max_val <= 0.0:
            return []

        peak_x, peak_y = max_loc
        peak_value = int(r_channel[peak_y, peak_x])
        if peak_value < self._config.laser_peak_min_value:
            ys, xs = np.where(seed_mask)
            if xs.size == 0:
                return []
            best_index = int(np.argmax(seed_response[ys, xs]))
            peak_x = int(xs[best_index])
            peak_y = int(ys[best_index])
            peak_value = int(r_channel[peak_y, peak_x])

        radius = int(self._config.laser_peak_window_radius_px)
        x1 = max(0, peak_x - radius)
        y1 = max(0, peak_y - radius)
        x2 = min(w, peak_x + radius + 1)
        y2 = min(h, peak_y + radius + 1)
        roi_r = r_channel[y1:y2, x1:x2]
        roi_dom = dominance[y1:y2, x1:x2]
        roi_sat = sat[y1:y2, x1:x2]
        roi_val = val[y1:y2, x1:x2]

        peak_threshold = max(self._config.laser_peak_min_value, int(peak_value * self._config.laser_peak_threshold_ratio))
        roi_mask = (
            (roi_r >= peak_threshold)
            & (roi_dom >= self._config.laser_red_delta)
            & (roi_sat >= self._config.laser_saturation_min)
            & (roi_val >= self._config.laser_value_min)
        ).astype(np.uint8) * 255
        if cv2.countNonZero(roi_mask) == 0:
            return []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, 8)
        candidates: list[LaserDotDetection] = []
        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 1.0 or area > self._config.laser_max_area_px:
                continue
            lx = int(stats[label, cv2.CC_STAT_LEFT])
            ly = int(stats[label, cv2.CC_STAT_TOP])
            lw = int(stats[label, cv2.CC_STAT_WIDTH])
            lh = int(stats[label, cv2.CC_STAT_HEIGHT])
            aspect = max(lw, lh) / max(1.0, min(lw, lh))
            if aspect > 2.2:
                continue
            component_mask = (labels == label).astype(np.uint8)
            ys, xs = np.where(component_mask > 0)
            if xs.size == 0:
                continue
            gx = xs + x1
            gy = ys + y1
            weights = roi_r[ys, xs].astype(np.float32) + (roi_dom[ys, xs].astype(np.float32) * 1.4)
            weights_sum = float(weights.sum())
            if weights_sum <= 0:
                continue
            cx = float((gx * weights).sum() / weights_sum)
            cy = float((gy * weights).sum() / weights_sum)
            area_contour = float(component_mask.sum())
            radius_px = float(np.sqrt(area_contour / np.pi))
            circularity = min(1.0, area_contour / max(np.pi * radius_px * radius_px, 1.0))
            mean_red = float(roi_r[ys, xs].mean())
            score = (mean_red * 1.6) + (weights_sum / max(1.0, area_contour) * 0.4) + (circularity * 120.0) - (area_contour * 0.8)
            if peak_x >= x1 + lx and peak_x < x1 + lx + lw and peak_y >= y1 + ly and peak_y < y1 + ly + lh:
                score += 180.0
            candidates.append(LaserDotDetection(
                center=(int(round(cx)), int(round(cy))),
                radius_px=max(1.0, radius_px),
                area_px=area_contour,
                score=float(score),
                circularity=float(circularity),
                mean_red=mean_red,
                bbox=(x1 + lx, y1 + ly, lw, lh),
            ))
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates


class PanTiltCalibrator:
    def __init__(self, config: PanTiltControlConfig, controller: PanTiltController, logger: logging.Logger, detector: LaserDotDetector | None = None) -> None:
        self._config = config
        self._controller = controller
        self._logger = logger
        self._detector = detector or LaserDotDetector(config)
        self._calibration: PanTiltCalibrationData | None = None
        self._progress = CalibrationProgress()
        self._pending_targets: list[tuple[float, float]] = []
        self._samples: list[PanTiltCalibrationSample] = []
        self._last_frame_shape: tuple[int, int] | None = None
        self._original_state: tuple[int, int, str, bool] | None = None
        self._current_target_started_at: float = 0.0
        self._current_target: tuple[float, float] | None = None
        self._reference_detection: LaserDotDetection | None = None
        self._last_good_detection: LaserDotDetection | None = None
        self._stable_detections: list[LaserDotDetection] = []

    @property
    def detector(self) -> LaserDotDetector:
        return self._detector

    @property
    def calibration(self) -> PanTiltCalibrationData | None:
        return self._calibration

    @property
    def progress(self) -> CalibrationProgress:
        return self._progress

    def load(self) -> PanTiltCalibrationData | None:
        path = Path(self._config.calibration_artifact_path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        self._calibration = PanTiltCalibrationData(
            version=int(payload.get('version', 1)),
            created_at=float(payload.get('created_at', time.time())),
            frame_width=int(payload['frame_width']),
            frame_height=int(payload['frame_height']),
            pan_coefficients=[float(v) for v in payload['pan_coefficients']],
            tilt_coefficients=[float(v) for v in payload['tilt_coefficients']],
            fit_error_degrees=float(payload.get('fit_error_degrees', 0.0)),
            samples=[PanTiltCalibrationSample(**sample) for sample in payload.get('samples', [])],
        )
        return self._calibration

    def start(self, frame: np.ndarray | None) -> str:
        if self._progress.running:
            return 'AUTO CAL already running'
        detection = self._detector.detect(frame)
        if detection is None:
            raise RuntimeError('Не вижу красную точку в текущем кадре. Наведи луч в поле зрения камеры и попробуй снова.')
        self._controller.maybe_refresh_state(force=True)
        state = self._controller.state
        self._original_state = (state.pan_angle, state.tilt_angle, state.speed_mode, state.laser_on)
        if not state.laser_on:
            self._controller.set_laser(True)
        if state.speed_mode != 'slow':
            self._controller.set_speed_mode('slow')
        self._samples = []
        self._last_frame_shape = frame.shape[:2]
        self._reference_detection = detection
        self._last_good_detection = detection
        self._stable_detections = []
        self._pending_targets = self._build_local_targets(float(state.pan_angle), float(state.tilt_angle))
        self._progress = CalibrationProgress(running=True, sample_index=0, total_samples=len(self._pending_targets), stage='moving', started_at=time.monotonic(), confirmed_frames=0)
        self._current_target = None
        self._current_target_started_at = 0.0
        return f'AUTO CAL started: {len(self._pending_targets)} local points'

    def tick(self, frame: np.ndarray | None) -> str | None:
        if not self._progress.running:
            return None
        if frame is not None and frame.size > 0:
            self._last_frame_shape = frame.shape[:2]
        now = time.monotonic()
        if self._current_target is None:
            if not self._pending_targets:
                return self._finish()
            self._current_target = self._pending_targets.pop(0)
            self._progress.sample_index += 1
            self._progress.current_target = self._current_target
            self._progress.stage = 'moving'
            self._progress.confirmed_frames = 0
            self._stable_detections = []
            self._controller.target_angles(int(round(self._current_target[0])), int(round(self._current_target[1])))
            self._current_target_started_at = now
            self._logger.info('Auto calibration sample %d -> pan=%s tilt=%s', self._progress.sample_index, self._current_target[0], self._current_target[1])
            return None

        if now - self._current_target_started_at < self._config.auto_calibration_settle_seconds:
            return None

        self._progress.stage = 'confirming'
        detection = self._detector.detect(
            frame,
            expected_center=self._last_good_detection.center if self._last_good_detection is not None else None,
            reference=self._reference_detection,
            max_distance_px=self._config.auto_calibration_max_jump_px,
        )
        if detection is not None:
            if self._stable_detections and _distance(detection.center, self._stable_detections[-1].center) > 80.0:
                self._stable_detections = [detection]
                self._progress.confirmed_frames = 1
            else:
                self._stable_detections.append(detection)
                self._progress.confirmed_frames += 1
        else:
            self._stable_detections.clear()
            self._progress.confirmed_frames = 0

        if self._progress.confirmed_frames < self._config.auto_calibration_confirm_frames:
            if now - self._current_target_started_at < self._config.auto_calibration_detection_timeout_seconds:
                return None
            self._logger.warning('Laser dot confirmation failed at pan=%s tilt=%s', self._current_target[0], self._current_target[1])
            self._current_target = None
            self._progress.current_target = None
            self._progress.stage = 'sampling'
            self._stable_detections.clear()
            self._progress.confirmed_frames = 0
            return None

        accepted = _median_detection(self._stable_detections)
        self._samples.append(PanTiltCalibrationSample(
            pan_angle=float(self._current_target[0]),
            tilt_angle=float(self._current_target[1]),
            pixel_x=float(accepted.center[0]),
            pixel_y=float(accepted.center[1]),
            radius_px=float(accepted.radius_px),
            area_px=float(accepted.area_px),
        ))
        self._last_good_detection = accepted
        self._current_target = None
        self._progress.current_target = None
        self._progress.stage = 'sampling'
        self._stable_detections.clear()
        self._progress.confirmed_frames = 0
        return None

    def stop(self) -> None:
        self._progress = CalibrationProgress()
        self._pending_targets = []
        self._current_target = None
        self._stable_detections = []
        self._restore_original_state()

    def aim_at_pixel(self, x: float, y: float) -> tuple[float, float]:
        calibration = self._calibration or self.load()
        if calibration is None:
            raise RuntimeError('PanTilt calibration is not loaded.')
        pan_angle, tilt_angle = calibration.pixel_to_angles(x, y)
        self._controller.target_angles(int(round(pan_angle)), int(round(tilt_angle)))
        return pan_angle, tilt_angle

    def _build_local_targets(self, pan_center: float, tilt_center: float) -> list[tuple[float, float]]:
        pan_half = max(0, self._config.auto_calibration_local_pan_points // 2)
        tilt_half = max(0, self._config.auto_calibration_local_tilt_points // 2)
        pan_values = [pan_center + (i * self._config.auto_calibration_local_pan_step_degrees) for i in range(-pan_half, pan_half + 1)]
        tilt_values = [tilt_center + (i * self._config.auto_calibration_local_tilt_step_degrees) for i in range(-tilt_half, tilt_half + 1)]
        pan_values = [min(self._config.auto_calibration_pan_max_angle, max(self._config.auto_calibration_pan_min_angle, v)) for v in pan_values]
        tilt_values = [min(self._config.auto_calibration_tilt_max_angle, max(self._config.auto_calibration_tilt_min_angle, v)) for v in tilt_values]
        grid: list[tuple[float, float]] = []
        seen: set[tuple[int, int]] = set()
        for row_index, tilt in enumerate(tilt_values):
            pan_iter = pan_values if row_index % 2 == 0 else list(reversed(pan_values))
            for pan in pan_iter:
                key = (int(round(pan * 10)), int(round(tilt * 10)))
                if key in seen:
                    continue
                seen.add(key)
                grid.append((float(pan), float(tilt)))
        return grid

    def _finish(self) -> str:
        minimum = max(6, self._config.auto_calibration_min_samples)
        if len(self._samples) < minimum:
            self._restore_original_state()
            self._progress = CalibrationProgress()
            raise RuntimeError(f'Calibration collected only {len(self._samples)} usable samples; need at least {minimum}.')
        if self._last_frame_shape is None:
            self._restore_original_state()
            self._progress = CalibrationProgress()
            raise RuntimeError('Calibration finished without frame shape.')
        h, w = self._last_frame_shape
        data = self._fit(samples=self._samples, frame_width=w, frame_height=h)
        if data.fit_error_degrees > self._config.auto_calibration_max_fit_error_degrees:
            self._restore_original_state()
            self._progress = CalibrationProgress()
            raise RuntimeError(f'Calibration fit too noisy: {data.fit_error_degrees:.1f} deg (need <= {self._config.auto_calibration_max_fit_error_degrees:.1f}).')
        self._calibration = data
        self._save(data)
        self._restore_original_state()
        self._progress = CalibrationProgress()
        return f'AUTO CAL ready: {len(data.samples)} pts, fit {data.fit_error_degrees:.1f} deg'

    def _restore_original_state(self) -> None:
        if self._original_state is None:
            return
        pan, tilt, speed, laser_on = self._original_state
        try:
            self._controller.target_angles(int(round(pan)), int(round(tilt)))
            self._controller.set_speed_mode(speed)
            self._controller.set_laser(laser_on)
        except Exception:
            self._logger.warning('Failed to restore original state after calibration.', exc_info=True)
        finally:
            self._original_state = None

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
            'version': data.version,
            'created_at': data.created_at,
            'frame_width': data.frame_width,
            'frame_height': data.frame_height,
            'pan_coefficients': data.pan_coefficients,
            'tilt_coefficients': data.tilt_coefficients,
            'fit_error_degrees': data.fit_error_degrees,
            'samples': [asdict(sample) for sample in data.samples],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _feature_vector(x: float, y: float, frame_width: int, frame_height: int) -> np.ndarray:
    width = max(1.0, float(frame_width))
    height = max(1.0, float(frame_height))
    nx = (float(x) / width) - 0.5
    ny = (float(y) / height) - 0.5
    return np.array([nx, ny, nx * ny, nx * nx, ny * ny, 1.0], dtype=np.float64)


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _median_detection(detections: list[LaserDotDetection]) -> LaserDotDetection:
    centers_x = sorted(item.center[0] for item in detections)
    centers_y = sorted(item.center[1] for item in detections)
    radii = sorted(item.radius_px for item in detections)
    areas = sorted(item.area_px for item in detections)
    scores = sorted(item.score for item in detections)
    circularities = sorted(item.circularity for item in detections)
    reds = sorted(item.mean_red for item in detections)
    mid = len(detections) // 2
    base = detections[mid]
    return LaserDotDetection(
        center=(int(centers_x[mid]), int(centers_y[mid])),
        radius_px=float(radii[mid]),
        area_px=float(areas[mid]),
        score=float(scores[mid]),
        circularity=float(circularities[mid]),
        mean_red=float(reds[mid]),
        bbox=base.bbox,
    )
