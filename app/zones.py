from __future__ import annotations

from dataclasses import dataclass

from app.models import AlertPointMode, FloatBBox, FrameShape, SceneZone, SceneZonesConfig, Track, ZoneType
from app.utils import bbox_area


@dataclass(frozen=True, slots=True)
class ZoneMatch:
    zone: SceneZone | None
    foot_point: tuple[float, float]
    foot_inside: bool
    overlap_ratio: float


@dataclass(frozen=True, slots=True)
class AlertZoneMatch:
    zone: SceneZone | None
    alert_point: tuple[float, float]
    point_inside: bool


@dataclass(frozen=True, slots=True)
class ProjectedZone:
    zone: SceneZone
    bounds: tuple[float, float, float, float]
    points: tuple[tuple[int, int], ...]

    @property
    def draw_points(self) -> tuple[tuple[int, int], ...]:
        if self.zone.shape_type == "rect":
            left, top, right, bottom = self.bounds
            return (
                (int(round(left)), int(round(top))),
                (int(round(right)), int(round(top))),
                (int(round(right)), int(round(bottom))),
                (int(round(left)), int(round(bottom))),
            )
        return self.points


class SceneZoneClassifier:
    def __init__(self, config: SceneZonesConfig) -> None:
        self._config = config

    @property
    def enabled_zones(self) -> list[SceneZone]:
        if not self._config.enabled:
            return []
        return [zone for zone in self._config.zones if zone.enabled]

    def classify_track(self, track: Track, frame_shape: FrameShape) -> ZoneMatch:
        foot_point = track_foot_point(track)
        candidates: list[tuple[tuple[float, float, float, float, str], SceneZone, bool, float]] = []

        for zone in self.enabled_zones:
            foot_inside = point_in_zone(zone, foot_point, frame_shape)
            overlap_ratio = bbox_overlap_ratio(track.bbox, zone, frame_shape)
            if not foot_inside and overlap_ratio < self._config.bbox_overlap_threshold:
                continue

            candidates.append(
                (
                    (
                        0.0 if foot_inside else 1.0,
                        float(_zone_type_rank(zone.zone_type, self._config.surface_priority_over_floor)),
                        -overlap_ratio,
                        zone_area(zone, frame_shape),
                        zone.name,
                    ),
                    zone,
                    foot_inside,
                    overlap_ratio,
                ),
            )

        if not candidates:
            return ZoneMatch(zone=None, foot_point=foot_point, foot_inside=False, overlap_ratio=0.0)

        _, zone, foot_inside, overlap_ratio = sorted(candidates, key=lambda item: item[0])[0]
        return ZoneMatch(
            zone=zone,
            foot_point=foot_point,
            foot_inside=foot_inside,
            overlap_ratio=overlap_ratio,
        )

    def classify_alert_track(
        self,
        track: Track,
        frame_shape: FrameShape,
        point_mode: AlertPointMode = "crosshair_center",
    ) -> AlertZoneMatch:
        alert_point = track_crosshair_point(track, point_mode=point_mode)
        return self.classify_point(alert_point, frame_shape)

    def classify_point(
        self,
        point: tuple[float, float],
        frame_shape: FrameShape,
    ) -> AlertZoneMatch:
        candidates: list[tuple[tuple[int, float, str], SceneZone]] = []
        for zone in self.enabled_zones:
            if not point_in_zone(zone, point, frame_shape):
                continue
            candidates.append(
                (
                    (
                        _zone_type_rank(zone.zone_type, self._config.surface_priority_over_floor),
                        zone_area(zone, frame_shape),
                        zone.name,
                    ),
                    zone,
                ),
            )

        if not candidates:
            return AlertZoneMatch(zone=None, alert_point=point, point_inside=False)

        _, zone = sorted(candidates, key=lambda item: item[0])[0]
        return AlertZoneMatch(zone=zone, alert_point=point, point_inside=True)


def track_foot_point(track: Track) -> tuple[float, float]:
    return ((track.bbox[0] + track.bbox[2]) / 2.0, track.bbox[3])


def track_crosshair_point(
    track: Track,
    point_mode: AlertPointMode = "crosshair_center",
) -> tuple[float, float]:
    if point_mode != "crosshair_center":
        raise ValueError(f"Unsupported alert point mode: {point_mode}")
    center_x, center_y = track.center
    return float(center_x), float(center_y)


def project_zone(zone: SceneZone, frame_shape: FrameShape) -> ProjectedZone:
    frame_height, frame_width = frame_shape[:2]
    if zone.shape_type == "rect":
        x1 = project_x(zone.x1 or 0.0, zone.coordinates_mode, frame_width)
        y1 = project_y(zone.y1 or 0.0, zone.coordinates_mode, frame_height)
        x2 = project_x(zone.x2 or 0.0, zone.coordinates_mode, frame_width)
        y2 = project_y(zone.y2 or 0.0, zone.coordinates_mode, frame_height)
        return ProjectedZone(
            zone=zone,
            bounds=(x1, y1, x2, y2),
            points=(),
        )

    points = tuple(
        (
            int(round(project_x(point_x, zone.coordinates_mode, frame_width))),
            int(round(project_y(point_y, zone.coordinates_mode, frame_height))),
        )
        for point_x, point_y in zone.points
    )
    xs = [point[0] for point in points] or [0]
    ys = [point[1] for point in points] or [0]
    return ProjectedZone(
        zone=zone,
        bounds=(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
        points=points,
    )


def project_x(value: float, coordinates_mode: str, frame_width: int) -> float:
    if coordinates_mode == "normalized":
        return max(0.0, min(1.0, value)) * frame_width
    return float(value)


def project_y(value: float, coordinates_mode: str, frame_height: int) -> float:
    if coordinates_mode == "normalized":
        return max(0.0, min(1.0, value)) * frame_height
    return float(value)


def normalize_x(value: float, frame_width: int) -> float:
    if frame_width <= 0:
        return 0.0
    return max(0.0, min(1.0, value / frame_width))


def normalize_y(value: float, frame_height: int) -> float:
    if frame_height <= 0:
        return 0.0
    return max(0.0, min(1.0, value / frame_height))


def normalize_point(point: tuple[float, float], frame_shape: FrameShape) -> tuple[float, float]:
    frame_height, frame_width = frame_shape[:2]
    return normalize_x(point[0], frame_width), normalize_y(point[1], frame_height)


def convert_zone_to_coordinates_mode(
    zone: SceneZone,
    frame_shape: FrameShape,
    target_mode: str,
) -> SceneZone:
    if zone.coordinates_mode == target_mode:
        return zone

    frame_height, frame_width = frame_shape[:2]
    if zone.shape_type == "rect":
        assert zone.x1 is not None and zone.y1 is not None and zone.x2 is not None and zone.y2 is not None
        if target_mode == "normalized":
            return SceneZone(
                name=zone.name,
                enabled=zone.enabled,
                zone_type=zone.zone_type,
                shape_type=zone.shape_type,
                coordinates_mode="normalized",
                color=zone.color,
                x1=normalize_x(zone.x1, frame_width),
                y1=normalize_y(zone.y1, frame_height),
                x2=normalize_x(zone.x2, frame_width),
                y2=normalize_y(zone.y2, frame_height),
            )
        return SceneZone(
            name=zone.name,
            enabled=zone.enabled,
            zone_type=zone.zone_type,
            shape_type=zone.shape_type,
            coordinates_mode="pixels",
            color=zone.color,
            x1=project_x(zone.x1, zone.coordinates_mode, frame_width),
            y1=project_y(zone.y1, zone.coordinates_mode, frame_height),
            x2=project_x(zone.x2, zone.coordinates_mode, frame_width),
            y2=project_y(zone.y2, zone.coordinates_mode, frame_height),
        )

    if target_mode == "normalized":
        points = tuple(normalize_point(point, frame_shape) for point in zone.points)
        return SceneZone(
            name=zone.name,
            enabled=zone.enabled,
            zone_type=zone.zone_type,
            shape_type=zone.shape_type,
            coordinates_mode="normalized",
            color=zone.color,
            points=points,
        )

    points = tuple(
        (
            project_x(point_x, zone.coordinates_mode, frame_width),
            project_y(point_y, zone.coordinates_mode, frame_height),
        )
        for point_x, point_y in zone.points
    )
    return SceneZone(
        name=zone.name,
        enabled=zone.enabled,
        zone_type=zone.zone_type,
        shape_type=zone.shape_type,
        coordinates_mode="pixels",
        color=zone.color,
        points=points,
    )


def point_in_zone(zone: SceneZone, point: tuple[float, float], frame_shape: FrameShape) -> bool:
    projected = project_zone(zone, frame_shape)
    x, y = point
    if zone.shape_type == "rect":
        left, top, right, bottom = projected.bounds
        return left <= x <= right and top <= y <= bottom
    return point_in_polygon(point, projected.points)


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[int, int], ...]) -> bool:
    x, y = point
    inside = False
    if len(polygon) < 3:
        return False

    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = ((current_y > y) != (previous_y > y)) and (
            x < ((previous_x - current_x) * (y - current_y) / ((previous_y - current_y) or 1e-9) + current_x)
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def zone_bounds(zone: SceneZone, frame_shape: FrameShape) -> tuple[float, float, float, float]:
    return project_zone(zone, frame_shape).bounds


def zone_area(zone: SceneZone, frame_shape: FrameShape) -> float:
    projected = project_zone(zone, frame_shape)
    if zone.shape_type == "rect":
        left, top, right, bottom = projected.bounds
        return max(0.0, right - left) * max(0.0, bottom - top)

    area = 0.0
    points = projected.points
    if len(points) < 3:
        return 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def bbox_overlap_ratio(track_bbox: FloatBBox, zone: SceneZone, frame_shape: FrameShape) -> float:
    zone_bbox = zone_bounds(zone, frame_shape)
    track_area = bbox_area(track_bbox)
    if track_area <= 0:
        return 0.0

    intersection_left = max(track_bbox[0], zone_bbox[0])
    intersection_top = max(track_bbox[1], zone_bbox[1])
    intersection_right = min(track_bbox[2], zone_bbox[2])
    intersection_bottom = min(track_bbox[3], zone_bbox[3])
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0:
        return 0.0
    return intersection_area / track_area


def pick_zone_at_point(
    zones: list[SceneZone],
    point: tuple[float, float],
    frame_shape: FrameShape,
) -> SceneZone | None:
    matches = [zone for zone in zones if zone.enabled and point_in_zone(zone, point, frame_shape)]
    if not matches:
        return None
    matches.sort(
        key=lambda zone: (
            _zone_type_rank(zone.zone_type, surface_priority_over_floor=True),
            zone_area(zone, frame_shape),
            zone.name,
        ),
    )
    return matches[0]


def _zone_type_rank(zone_type: ZoneType, surface_priority_over_floor: bool) -> int:
    if not surface_priority_over_floor:
        return 0
    priorities = {
        ZoneType.RESTRICTED: 0,
        ZoneType.SURFACE: 1,
        ZoneType.FLOOR: 2,
    }
    return priorities[zone_type]
