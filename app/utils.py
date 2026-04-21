from __future__ import annotations

import math
import time
from dataclasses import dataclass

from app.models import BBox, Detection, DetectionOverlay, FloatBBox, FrameShape, Track


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> BBox:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    clamped_x1 = int(round(clamp(x1, 0, frame_width - 1)))
    clamped_y1 = int(round(clamp(y1, 0, frame_height - 1)))
    clamped_x2 = int(round(clamp(x2, clamped_x1 + 1, frame_width)))
    clamped_y2 = int(round(clamp(y2, clamped_y1 + 1, frame_height)))
    return clamped_x1, clamped_y1, clamped_x2, clamped_y2


def clamp_float_bbox(
    bbox: FloatBBox,
    frame_width: int,
    frame_height: int,
) -> FloatBBox:
    x1, y1, x2, y2 = bbox
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    clamped_x1 = clamp(x1, 0.0, max(frame_width - 1.0, 0.0))
    clamped_y1 = clamp(y1, 0.0, max(frame_height - 1.0, 0.0))
    clamped_x2 = clamp(x2, clamped_x1 + 1.0, float(frame_width))
    clamped_y2 = clamp(y2, clamped_y1 + 1.0, float(frame_height))
    return clamped_x1, clamped_y1, clamped_x2, clamped_y2


def square_from_center(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> BBox:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if side <= 0:
        raise ValueError("Square side must be positive.")

    side_int = int(round(min(side, frame_width, frame_height)))
    side_int = max(1, side_int)

    max_left = max(frame_width - side_int, 0)
    max_top = max(frame_height - side_int, 0)
    left = int(round(clamp(center_x - (side_int / 2.0), 0, max_left)))
    top = int(round(clamp(center_y - (side_int / 2.0), 0, max_top)))
    return left, top, left + side_int, top + side_int


def bbox_to_square(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> BBox:
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box must have positive width and height.")
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(width, height)
    return square_from_center(center_x, center_y, side, frame_width, frame_height)


def calculate_center_from_bbox(bbox: BBox | FloatBBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def normalize_coordinates(
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    return center_x / frame_width, center_y / frame_height


def bbox_area(bbox: BBox | FloatBBox) -> float:
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return width * height


def bbox_iou(first_bbox: BBox | FloatBBox, second_bbox: BBox | FloatBBox) -> float:
    intersection_left = max(first_bbox[0], second_bbox[0])
    intersection_top = max(first_bbox[1], second_bbox[1])
    intersection_right = min(first_bbox[2], second_bbox[2])
    intersection_bottom = min(first_bbox[3], second_bbox[3])

    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0:
        return 0.0

    first_area = bbox_area(first_bbox)
    second_area = bbox_area(second_bbox)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def center_distance(
    first_center: tuple[float, float],
    second_center: tuple[float, float],
) -> float:
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def area_ratio(first_area: float, second_area: float) -> float:
    if first_area <= 0 or second_area <= 0:
        return 0.0
    return first_area / second_area


def smooth_value(previous_value: float, current_value: float, alpha: float) -> float:
    return ((1.0 - alpha) * previous_value) + (alpha * current_value)


def smooth_bbox(previous_bbox: FloatBBox, current_bbox: FloatBBox, alpha: float) -> FloatBBox:
    return (
        smooth_value(previous_bbox[0], current_bbox[0], alpha),
        smooth_value(previous_bbox[1], current_bbox[1], alpha),
        smooth_value(previous_bbox[2], current_bbox[2], alpha),
        smooth_value(previous_bbox[3], current_bbox[3], alpha),
    )


def sort_tracks_for_display(
    tracks: list[Track],
    mode: str = "top_to_bottom_left_to_right",
) -> list[Track]:
    if mode != "top_to_bottom_left_to_right":
        raise ValueError(f"Unsupported display sort mode: {mode}")
    return sorted(
        tracks,
        key=lambda track: (
            round(track.center_y, 3),
            round(track.center_x, 3),
            track.track_id,
        ),
    )


def sort_detections_for_display(detections: list[Detection]) -> list[Detection]:
    return sorted(
        detections,
        key=lambda detection: (
            round(detection.center_y, 3),
            round(detection.center_x, 3),
            -detection.confidence,
        ),
    )


def build_detection_overlays(
    detections: list[Detection],
    frame_shape: FrameShape,
) -> list[DetectionOverlay]:
    frame_height, frame_width = frame_shape[:2]
    overlays: list[DetectionOverlay] = []
    for display_number, detection in enumerate(sort_detections_for_display(detections), start=1):
        square_bbox = bbox_to_square(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            frame_width,
            frame_height,
        )
        normalized_x, normalized_y = normalize_coordinates(
            detection.center_x,
            detection.center_y,
            frame_width,
            frame_height,
        )
        overlays.append(
            DetectionOverlay(
                display_number=display_number,
                square_bbox=square_bbox,
                center_x=detection.center_x,
                center_y=detection.center_y,
                normalized_x=normalized_x,
                normalized_y=normalized_y,
                confidence=detection.confidence,
                class_name=detection.class_name,
            ),
        )
    return overlays


def safe_float_text(value: float | None, precision: int = 3, default: str = "n/a") -> str:
    if value is None:
        return default
    return f"{value:.{precision}f}"


@dataclass(slots=True)
class FPSMeter:
    smoothing: float = 0.9
    _last_timestamp: float | None = None
    _fps: float = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_timestamp is None:
            self._last_timestamp = now
            return self._fps

        delta = now - self._last_timestamp
        self._last_timestamp = now
        if delta <= 0:
            return self._fps

        instant_fps = 1.0 / delta
        if self._fps <= 0:
            self._fps = instant_fps
        else:
            self._fps = (self.smoothing * self._fps) + ((1.0 - self.smoothing) * instant_fps)
        return self._fps
