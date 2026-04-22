from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from app.models import (
    AlertRecordingConfig,
    AppConfig,
    DetectorConfig,
    LoggingConfig,
    OutputConfig,
    OverlayConfig,
    ResizeConfig,
    SceneZone,
    SceneZonesConfig,
    SourceConfig,
    SurfaceAlertConfig,
    TrackingConfig,
    ZoneType,
    ZoneCoordinatesMode,
)


class ConfigError(RuntimeError):
    """Raised when application configuration is invalid."""


_DEFAULT_CONFIG = AppConfig()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Cannot parse boolean value from '{value}'.")


def load_config(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    raw_config = load_raw_config(config_path)
    if overrides:
        _apply_overrides(raw_config, overrides)
    config = _build_app_config(raw_config)
    _validate_config(config)
    return config


def load_raw_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file: {path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Root config structure must be a mapping.")
    return data


def save_scene_zones_config(config_path: str, scene_zones: SceneZonesConfig) -> None:
    path = Path(config_path)
    raw_config = load_raw_config(str(path)) if path.exists() else {}
    raw_config["scene_zones"] = serialize_scene_zones_config(scene_zones)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(raw_config, file, sort_keys=False, allow_unicode=True)


def serialize_scene_zones_config(scene_zones: SceneZonesConfig) -> dict[str, Any]:
    return {
        "enabled": scene_zones.enabled,
        "draw_zones": scene_zones.draw_zones,
        "draw_track_locations": scene_zones.draw_track_locations,
        "surface_priority_over_floor": scene_zones.surface_priority_over_floor,
        "bbox_overlap_threshold": _round_float(scene_zones.bbox_overlap_threshold),
        "coordinates_mode": scene_zones.coordinates_mode,
        "zone_editor_enabled": scene_zones.zone_editor_enabled,
        "zones": [serialize_scene_zone(zone) for zone in scene_zones.zones],
    }


def serialize_scene_zone(zone: SceneZone) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": zone.name,
        "enabled": zone.enabled,
        "zone_type": zone.zone_type.value,
        "shape_type": zone.shape_type,
        "coordinates_mode": zone.coordinates_mode,
    }
    if zone.color is not None:
        payload["color"] = list(zone.color)

    if zone.shape_type == "rect":
        payload["x1"] = _round_float(zone.x1 or 0.0)
        payload["y1"] = _round_float(zone.y1 or 0.0)
        payload["x2"] = _round_float(zone.x2 or 0.0)
        payload["y2"] = _round_float(zone.y2 or 0.0)
    else:
        payload["points"] = [
            [_round_float(point_x), _round_float(point_y)]
            for point_x, point_y in zone.points
        ]
    return payload


def _build_app_config(data: Mapping[str, Any]) -> AppConfig:
    defaults = _DEFAULT_CONFIG
    source_data = _mapping(data.get("source"))
    detector_data = _mapping(data.get("detector"))
    overlay_data = _mapping(data.get("overlay"))
    output_data = _mapping(data.get("output"))
    logging_data = _mapping(data.get("logging"))
    tracking_data = _mapping(data.get("tracking"))
    scene_zones_data = _mapping(data.get("scene_zones"))
    surface_alert_data = _mapping(data.get("surface_alert"))
    alert_recording_data = _mapping(data.get("alert_recording"))
    resize_data = _mapping(data.get("resize"))

    overlay = OverlayConfig(
        minimal_overlay=bool(overlay_data.get("minimal_overlay", defaults.overlay.minimal_overlay)),
        debug_overlay=bool(overlay_data.get("debug_overlay", defaults.overlay.debug_overlay)),
        show_fps=bool(overlay_data.get("show_fps", defaults.overlay.show_fps)),
        show_debug_counters=bool(
            overlay_data.get("show_debug_counters", defaults.overlay.show_debug_counters),
        ),
        show_track_boxes=bool(
            overlay_data.get("show_track_boxes", defaults.overlay.show_track_boxes),
        ),
        show_track_crosshair=bool(
            overlay_data.get("show_track_crosshair", defaults.overlay.show_track_crosshair),
        ),
        show_track_labels=bool(
            overlay_data.get("show_track_labels", defaults.overlay.show_track_labels),
        ),
        show_detection_labels=bool(
            overlay_data.get("show_detection_labels", defaults.overlay.show_detection_labels),
        ),
        show_zone_labels=bool(
            overlay_data.get("show_zone_labels", defaults.overlay.show_zone_labels),
        ),
        show_cat_count=bool(
            overlay_data.get("show_cat_count", defaults.overlay.show_cat_count),
        ),
        box_color=_coerce_color(overlay_data.get("box_color"), defaults.overlay.box_color),
        predicted_box_color=_coerce_color(
            overlay_data.get("predicted_box_color"),
            defaults.overlay.predicted_box_color,
        ),
        secondary_box_color=_coerce_color(
            overlay_data.get("secondary_box_color"),
            defaults.overlay.secondary_box_color,
        ),
        marker_color=_coerce_color(overlay_data.get("marker_color"), defaults.overlay.marker_color),
        marker_crosshair_color=_coerce_color(
            overlay_data.get("marker_crosshair_color"),
            defaults.overlay.marker_crosshair_color,
        ),
        marker_dot_color=_coerce_color(
            overlay_data.get("marker_dot_color"),
            defaults.overlay.marker_dot_color,
        ),
        text_color=_coerce_color(overlay_data.get("text_color"), defaults.overlay.text_color),
        accent_color=_coerce_color(overlay_data.get("accent_color"), defaults.overlay.accent_color),
        warning_color=_coerce_color(overlay_data.get("warning_color"), defaults.overlay.warning_color),
        error_color=_coerce_color(overlay_data.get("error_color"), defaults.overlay.error_color),
        background_color=_coerce_color(
            overlay_data.get("background_color"),
            defaults.overlay.background_color,
        ),
        box_thickness=int(overlay_data.get("box_thickness", defaults.overlay.box_thickness)),
        secondary_box_thickness=int(
            overlay_data.get("secondary_box_thickness", defaults.overlay.secondary_box_thickness),
        ),
        marker_thickness=int(overlay_data.get("marker_thickness", defaults.overlay.marker_thickness)),
        marker_size=int(overlay_data.get("marker_size", defaults.overlay.marker_size)),
        marker_gap=int(overlay_data.get("marker_gap", defaults.overlay.marker_gap)),
        center_radius=int(overlay_data.get("center_radius", defaults.overlay.center_radius)),
        marker_ring_radius=int(overlay_data.get("marker_ring_radius", defaults.overlay.marker_ring_radius)),
        marker_dot_radius=int(overlay_data.get("marker_dot_radius", defaults.overlay.marker_dot_radius)),
        font_scale=float(overlay_data.get("font_scale", defaults.overlay.font_scale)),
        font_thickness=int(overlay_data.get("font_thickness", defaults.overlay.font_thickness)),
        padding=int(overlay_data.get("padding", defaults.overlay.padding)),
        line_height=int(overlay_data.get("line_height", defaults.overlay.line_height)),
        show_normalized_coords=bool(
            overlay_data.get("show_normalized_coords", defaults.overlay.show_normalized_coords),
        ),
    )

    return AppConfig(
        source=SourceConfig(
            source_type=str(source_data.get("source_type", defaults.source.source_type)),
            source_path=source_data.get("source_path", defaults.source.source_path),
            camera_index=int(source_data.get("camera_index", defaults.source.camera_index)),
            camera_width=(
                int(source_data["camera_width"])
                if source_data.get("camera_width") is not None
                else defaults.source.camera_width
            ),
            camera_height=(
                int(source_data["camera_height"])
                if source_data.get("camera_height") is not None
                else defaults.source.camera_height
            ),
            process_every_n_frames=int(
                source_data.get("process_every_n_frames", defaults.source.process_every_n_frames),
            ),
            stream_url=source_data.get("stream_url", defaults.source.stream_url),
            reconnect_delay_seconds=float(
                source_data.get("reconnect_delay_seconds", defaults.source.reconnect_delay_seconds),
            ),
            max_reconnect_attempts=int(
                source_data.get("max_reconnect_attempts", defaults.source.max_reconnect_attempts),
            ),
            read_fail_threshold=int(source_data.get("read_fail_threshold", defaults.source.read_fail_threshold)),
            buffer_size=int(source_data.get("buffer_size", defaults.source.buffer_size)),
            snapshot_timeout_seconds=float(
                source_data.get("snapshot_timeout_seconds", defaults.source.snapshot_timeout_seconds),
            ),
            snapshot_use_cache_bust=bool(
                source_data.get("snapshot_use_cache_bust", defaults.source.snapshot_use_cache_bust),
            ),
        ),
        detector=DetectorConfig(
            model_path=str(detector_data.get("model_path", defaults.detector.model_path)),
            imgsz=int(detector_data.get("imgsz", defaults.detector.imgsz)),
            confidence_threshold=float(
                detector_data.get("confidence_threshold", defaults.detector.confidence_threshold),
            ),
            iou_threshold=float(detector_data.get("iou_threshold", defaults.detector.iou_threshold)),
            device=str(detector_data.get("device", defaults.detector.device)),
            class_name=str(detector_data.get("class_name", defaults.detector.class_name)),
            max_frame_area_ratio=float(
                detector_data.get("max_frame_area_ratio", defaults.detector.max_frame_area_ratio),
            ),
        ),
        target_selection_strategy=str(
            data.get("target_selection_strategy", defaults.target_selection_strategy),
        ),
        overlay=overlay,
        output=OutputConfig(
            show_window=bool(output_data.get("show_window", defaults.output.show_window)),
            save_output=bool(output_data.get("save_output", defaults.output.save_output)),
            output_path=str(output_data.get("output_path", defaults.output.output_path)),
            window_name=str(output_data.get("window_name", defaults.output.window_name)),
        ),
        logging=LoggingConfig(
            log_level=str(logging_data.get("log_level", defaults.logging.log_level)).upper(),
            per_frame_debug=bool(
                _coalesce(
                    logging_data.get("per_frame_debug"),
                    logging_data.get("per_frame_debug_logging"),
                    default=defaults.logging.per_frame_debug,
                ),
            ),
            log_coordinates=bool(
                _coalesce(
                    logging_data.get("log_coordinates"),
                    output_data.get("log_coordinates"),
                    default=defaults.logging.log_coordinates,
                ),
            ),
            coordinates_log_path=_coalesce(
                logging_data.get("coordinates_log_path"),
                output_data.get("coordinates_log_path"),
                default=defaults.logging.coordinates_log_path,
            ),
            coordinates_log_format=str(
                _coalesce(
                    logging_data.get("coordinates_log_format"),
                    output_data.get("coordinates_log_format"),
                    default=defaults.logging.coordinates_log_format,
                ),
            ).lower(),
        ),
        tracking=TrackingConfig(
            tracking_enabled=bool(tracking_data.get("tracking_enabled", defaults.tracking.tracking_enabled)),
            multi_target_enabled=bool(
                tracking_data.get("multi_target_enabled", defaults.tracking.multi_target_enabled),
            ),
            tracker_only_mode_after_confirm=bool(
                tracking_data.get(
                    "tracker_only_mode_after_confirm",
                    defaults.tracking.tracker_only_mode_after_confirm,
                ),
            ),
            detector_interval_while_tracking=int(
                tracking_data.get(
                    "detector_interval_while_tracking",
                    defaults.tracking.detector_interval_while_tracking,
                ),
            ),
            reacquire_after_failed_tracker_frames=int(
                tracking_data.get(
                    "reacquire_after_failed_tracker_frames",
                    defaults.tracking.reacquire_after_failed_tracker_frames,
                ),
            ),
            max_tracker_only_frames_without_detection=int(
                tracking_data.get(
                    "max_tracker_only_frames_without_detection",
                    defaults.tracking.max_tracker_only_frames_without_detection,
                ),
            ),
            acquire_confidence_threshold=float(
                tracking_data.get(
                    "acquire_confidence_threshold",
                    tracking_data.get("min_confirm_confidence", defaults.tracking.acquire_confidence_threshold),
                ),
            ),
            keep_confidence_threshold=float(
                tracking_data.get("keep_confidence_threshold", defaults.tracking.keep_confidence_threshold),
            ),
            confirm_frames=int(tracking_data.get("confirm_frames", defaults.tracking.confirm_frames)),
            reconfirm_frames=int(tracking_data.get("reconfirm_frames", defaults.tracking.reconfirm_frames)),
            hold_without_detection_frames=int(
                tracking_data.get(
                    "hold_without_detection_frames",
                    defaults.tracking.hold_without_detection_frames,
                ),
            ),
            lost_transition_frames=int(
                tracking_data.get("lost_transition_frames", defaults.tracking.lost_transition_frames),
            ),
            hard_remove_frames=int(tracking_data.get("hard_remove_frames", defaults.tracking.hard_remove_frames)),
            min_confirm_confidence=float(
                tracking_data.get(
                    "acquire_confidence_threshold",
                    tracking_data.get("min_confirm_confidence", defaults.tracking.min_confirm_confidence),
                ),
            ),
            max_missing_frames=int(
                tracking_data.get(
                    "max_missing_frames",
                    tracking_data.get("max_lost_frames", defaults.tracking.max_missing_frames),
                ),
            ),
            reacquire_max_frames=int(
                tracking_data.get("reacquire_max_frames", defaults.tracking.reacquire_max_frames),
            ),
            iou_gate=float(tracking_data.get("iou_gate", defaults.tracking.iou_gate)),
            center_distance_gate=float(
                tracking_data.get("center_distance_gate", defaults.tracking.center_distance_gate),
            ),
            min_area_ratio=float(tracking_data.get("min_area_ratio", defaults.tracking.min_area_ratio)),
            max_area_ratio=float(tracking_data.get("max_area_ratio", defaults.tracking.max_area_ratio)),
            soft_iou_gate=float(tracking_data.get("soft_iou_gate", defaults.tracking.soft_iou_gate)),
            soft_center_distance_gate=float(
                tracking_data.get(
                    "soft_center_distance_gate",
                    defaults.tracking.soft_center_distance_gate,
                ),
            ),
            soft_min_area_ratio=float(
                tracking_data.get("soft_min_area_ratio", defaults.tracking.soft_min_area_ratio),
            ),
            soft_max_area_ratio=float(
                tracking_data.get("soft_max_area_ratio", defaults.tracking.soft_max_area_ratio),
            ),
            smoothing_alpha=float(tracking_data.get("smoothing_alpha", defaults.tracking.smoothing_alpha)),
            preserve_confirmed_tracks=bool(
                tracking_data.get("preserve_confirmed_tracks", defaults.tracking.preserve_confirmed_tracks),
            ),
            never_drop_confirmed_on_single_bad_frame=bool(
                tracking_data.get(
                    "never_drop_confirmed_on_single_bad_frame",
                    defaults.tracking.never_drop_confirmed_on_single_bad_frame,
                ),
            ),
            keep_lost_tracks=bool(tracking_data.get("keep_lost_tracks", defaults.tracking.keep_lost_tracks)),
            max_active_tracks=int(tracking_data.get("max_active_tracks", defaults.tracking.max_active_tracks)),
            use_motion_prediction=bool(
                tracking_data.get("use_motion_prediction", defaults.tracking.use_motion_prediction),
            ),
            display_sort_mode=str(
                tracking_data.get("display_sort_mode", defaults.tracking.display_sort_mode),
            ),
        ),
        scene_zones=_build_scene_zones_config(scene_zones_data, defaults),
        surface_alert=_build_surface_alert_config(surface_alert_data, defaults),
        alert_recording=_build_alert_recording_config(alert_recording_data, defaults),
        resize=ResizeConfig(
            enabled=bool(resize_data.get("enabled", defaults.resize.enabled)),
            width=int(resize_data["width"]) if resize_data.get("width") is not None else defaults.resize.width,
            height=int(resize_data["height"]) if resize_data.get("height") is not None else defaults.resize.height,
        ),
    )


def _build_scene_zones_config(scene_zones_data: Mapping[str, Any], defaults: AppConfig) -> SceneZonesConfig:
    coordinates_mode = _coerce_coordinates_mode(
        scene_zones_data.get("coordinates_mode"),
        default=defaults.scene_zones.coordinates_mode,
    )
    zones_data = scene_zones_data.get("zones", defaults.scene_zones.zones)
    if zones_data is None:
        zones_data = []
    if not isinstance(zones_data, list):
        raise ConfigError("scene_zones.zones must be a list.")

    zones = [_build_scene_zone(zone_data, default_coordinates_mode=coordinates_mode) for zone_data in zones_data]
    return SceneZonesConfig(
        enabled=bool(scene_zones_data.get("enabled", defaults.scene_zones.enabled)),
        draw_zones=bool(scene_zones_data.get("draw_zones", defaults.scene_zones.draw_zones)),
        draw_track_locations=bool(
            scene_zones_data.get("draw_track_locations", defaults.scene_zones.draw_track_locations),
        ),
        surface_priority_over_floor=bool(
            scene_zones_data.get(
                "surface_priority_over_floor",
                defaults.scene_zones.surface_priority_over_floor,
            ),
        ),
        bbox_overlap_threshold=float(
            scene_zones_data.get("bbox_overlap_threshold", defaults.scene_zones.bbox_overlap_threshold),
        ),
        coordinates_mode=coordinates_mode,
        zone_editor_enabled=bool(
            scene_zones_data.get("zone_editor_enabled", defaults.scene_zones.zone_editor_enabled),
        ),
        zones=zones,
    )


def _build_scene_zone(zone_data: Any, default_coordinates_mode: ZoneCoordinatesMode) -> SceneZone:
    if not isinstance(zone_data, Mapping):
        raise ConfigError("Each scene_zones.zones entry must be a mapping.")

    name = str(zone_data.get("name", "")).strip()
    zone_type_text = str(zone_data.get("zone_type", "")).strip().lower()
    shape_type = str(zone_data.get("shape_type", "")).strip().lower()
    if not name:
        raise ConfigError("scene_zones.zones entries must have a non-empty name.")
    try:
        zone_type = ZoneType(zone_type_text)
    except ValueError as exc:
        raise ConfigError(f"scene zone '{name}' has invalid zone_type '{zone_type_text}'.") from exc
    if shape_type not in {"rect", "polygon"}:
        raise ConfigError(f"scene zone '{name}' has invalid shape_type '{shape_type}'.")

    coordinates_mode = _infer_zone_coordinates_mode(zone_data, default_coordinates_mode)
    color = _coerce_optional_color(zone_data.get("color"))

    points: tuple[tuple[float, float], ...] = ()
    if shape_type == "polygon":
        raw_points = zone_data.get("points", [])
        if not isinstance(raw_points, list):
            raise ConfigError(f"scene zone '{name}' points must be a list.")
        points = tuple(_coerce_point(point, zone_name=name) for point in raw_points)

    return SceneZone(
        name=name,
        enabled=bool(zone_data.get("enabled", True)),
        zone_type=zone_type,
        shape_type=shape_type,
        coordinates_mode=coordinates_mode,
        color=color,
        x1=_coerce_optional_float(zone_data.get("x1")),
        y1=_coerce_optional_float(zone_data.get("y1")),
        x2=_coerce_optional_float(zone_data.get("x2")),
        y2=_coerce_optional_float(zone_data.get("y2")),
        points=points,
    )


def _build_surface_alert_config(surface_alert_data: Mapping[str, Any], defaults: AppConfig) -> SurfaceAlertConfig:
    raw_continuous_zone_types = surface_alert_data.get(
        "continuous_zone_types",
        defaults.surface_alert.continuous_zone_types,
    )
    if not isinstance(raw_continuous_zone_types, (list, tuple)):
        raise ConfigError("surface_alert.continuous_zone_types must be a list or tuple.")

    continuous_zone_types = tuple(str(item).strip().lower() for item in raw_continuous_zone_types)

    return SurfaceAlertConfig(
        enabled=bool(surface_alert_data.get("enabled", defaults.surface_alert.enabled)),
        trigger_on_surface_entry=bool(
            surface_alert_data.get(
                "trigger_on_surface_entry",
                defaults.surface_alert.trigger_on_surface_entry,
            ),
        ),
        alert_point_mode=str(
            surface_alert_data.get(
                "alert_point_mode",
                defaults.surface_alert.alert_point_mode,
            ),
        ),
        trigger_only_from_floor=bool(
            surface_alert_data.get(
                "trigger_only_from_floor",
                defaults.surface_alert.trigger_only_from_floor,
            ),
        ),
        trigger_from_unknown=bool(
            surface_alert_data.get("trigger_from_unknown", defaults.surface_alert.trigger_from_unknown),
        ),
        cooldown_seconds=float(
            surface_alert_data.get("cooldown_seconds", defaults.surface_alert.cooldown_seconds),
        ),
        min_interval_per_track=float(
            surface_alert_data.get("min_interval_per_track", defaults.surface_alert.min_interval_per_track),
        ),
        global_min_interval=float(
            surface_alert_data.get("global_min_interval", defaults.surface_alert.global_min_interval),
        ),
        repeat_on_same_surface=bool(
            surface_alert_data.get("repeat_on_same_surface", defaults.surface_alert.repeat_on_same_surface),
        ),
        sound_file=surface_alert_data.get("sound_file", defaults.surface_alert.sound_file),
        beep_fallback=bool(surface_alert_data.get("beep_fallback", defaults.surface_alert.beep_fallback)),

        continuous_while_in_zone=bool(
            surface_alert_data.get(
                "continuous_while_in_zone",
                defaults.surface_alert.continuous_while_in_zone,
            ),
        ),
        continuous_zone_types=continuous_zone_types,
        repeat_interval_seconds=float(
            surface_alert_data.get(
                "repeat_interval_seconds",
                defaults.surface_alert.repeat_interval_seconds,
            ),
        ),
        stop_when_zone_empty=bool(
            surface_alert_data.get(
                "stop_when_zone_empty",
                defaults.surface_alert.stop_when_zone_empty,
            ),
        ),

        show_overlay_message=bool(
            surface_alert_data.get(
                "show_overlay_message",
                defaults.surface_alert.show_overlay_message,
            ),
        ),
        overlay_message_frames=int(
            surface_alert_data.get(
                "overlay_message_frames",
                defaults.surface_alert.overlay_message_frames,
            ),
        ),
    )


def _build_alert_recording_config(
    alert_recording_data: Mapping[str, Any],
    defaults: AppConfig,
) -> AlertRecordingConfig:
    return AlertRecordingConfig(
        enabled=bool(alert_recording_data.get("enabled", defaults.alert_recording.enabled)),
        output_dir=str(alert_recording_data.get("output_dir", defaults.alert_recording.output_dir)),
        prebuffer_seconds=float(
            alert_recording_data.get(
                "prebuffer_seconds",
                defaults.alert_recording.prebuffer_seconds,
            ),
        ),
        postbuffer_seconds=float(
            alert_recording_data.get(
                "postbuffer_seconds",
                defaults.alert_recording.postbuffer_seconds,
            ),
        ),
        fps_fallback=float(
            alert_recording_data.get("fps_fallback", defaults.alert_recording.fps_fallback),
        ),
        codec=str(alert_recording_data.get("codec", defaults.alert_recording.codec)),
        draw_recording_overlay=bool(
            alert_recording_data.get(
                "draw_recording_overlay",
                defaults.alert_recording.draw_recording_overlay,
            ),
        ),
        include_wallclock_timestamp=bool(
            alert_recording_data.get(
                "include_wallclock_timestamp",
                defaults.alert_recording.include_wallclock_timestamp,
            ),
        ),
        include_zone_name=bool(
            alert_recording_data.get(
                "include_zone_name",
                defaults.alert_recording.include_zone_name,
            ),
        ),
        include_track_ids=bool(
            alert_recording_data.get(
                "include_track_ids",
                defaults.alert_recording.include_track_ids,
            ),
        ),
        include_elapsed_seconds=bool(
            alert_recording_data.get(
                "include_elapsed_seconds",
                defaults.alert_recording.include_elapsed_seconds,
            ),
        ),
    )


def _apply_overrides(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        key_parts = dotted_key.split(".")
        cursor = target
        for part in key_parts[:-1]:
            next_value = cursor.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[part] = next_value
            cursor = next_value
        cursor[key_parts[-1]] = value


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("Nested config sections must be mappings.")
    return value


def _coalesce(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _coerce_color(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ConfigError("Color values must be lists or tuples of length 3.")
    return int(value[0]), int(value[1]), int(value[2])


def _coerce_optional_color(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ConfigError("Zone color values must be lists or tuples of length 3.")
    return int(value[0]), int(value[1]), int(value[2])


def _coerce_point(point: Any, zone_name: str) -> tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ConfigError(f"scene zone '{zone_name}' polygon points must be [x, y] pairs.")
    return float(point[0]), float(point[1])


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_coordinates_mode(value: Any, default: ZoneCoordinatesMode) -> ZoneCoordinatesMode:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized not in {"normalized", "pixels"}:
        raise ConfigError(f"Invalid coordinates_mode '{value}'.")
    return normalized  # type: ignore[return-value]


def _infer_zone_coordinates_mode(
    zone_data: Mapping[str, Any],
    default_coordinates_mode: ZoneCoordinatesMode,
) -> ZoneCoordinatesMode:
    explicit = zone_data.get("coordinates_mode")
    if explicit is not None:
        return _coerce_coordinates_mode(explicit, default_coordinates_mode)

    numeric_values: list[float] = []
    for key in ("x1", "y1", "x2", "y2"):
        if zone_data.get(key) is not None:
            numeric_values.append(float(zone_data[key]))
    raw_points = zone_data.get("points")
    if isinstance(raw_points, list):
        for point in raw_points:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                numeric_values.extend((float(point[0]), float(point[1])))

    if numeric_values:
        if any(value > 1.0 for value in numeric_values):
            return "pixels"
        if all(0.0 <= value <= 1.0 for value in numeric_values):
            return "normalized"
    return default_coordinates_mode


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _validate_config(config: AppConfig) -> None:
    allowed_sources = {"webcam", "file", "rtsp", "http_snapshot"}
    if config.source.source_type not in allowed_sources:
        raise ConfigError(
            f"Unsupported source_type '{config.source.source_type}'. Expected one of {sorted(allowed_sources)}."
        )

    allowed_strategies = {"largest_area", "highest_confidence", "closest_to_center"}
    if config.target_selection_strategy not in allowed_strategies:
        raise ConfigError(f"Invalid target_selection_strategy: {config.target_selection_strategy}")

    if config.source.source_type == "file" and not config.source.source_path:
        raise ConfigError("source.source_path is required for file input.")
    if config.source.source_type in {"rtsp", "http_snapshot"} and not config.source.stream_url:
        raise ConfigError(f"source.stream_url is required for {config.source.source_type} input.")

    if not 0.0 <= config.detector.confidence_threshold <= 1.0:
        raise ConfigError("detector.confidence_threshold must be between 0 and 1.")
    if not 0.0 <= config.detector.iou_threshold <= 1.0:
        raise ConfigError("detector.iou_threshold must be between 0 and 1.")
    if config.detector.imgsz <= 0:
        raise ConfigError("detector.imgsz must be >= 1.")
    if not 0.0 < config.detector.max_frame_area_ratio <= 1.0:
        raise ConfigError("detector.max_frame_area_ratio must be between 0 and 1.")
    if config.source.camera_width is not None and config.source.camera_width <= 0:
        raise ConfigError("source.camera_width must be positive when set.")
    if config.source.camera_height is not None and config.source.camera_height <= 0:
        raise ConfigError("source.camera_height must be positive when set.")
    if config.source.process_every_n_frames <= 0:
        raise ConfigError("source.process_every_n_frames must be >= 1.")
    if config.source.snapshot_timeout_seconds <= 0:
        raise ConfigError("source.snapshot_timeout_seconds must be > 0.")
    if config.tracking.detector_interval_while_tracking <= 0:
        raise ConfigError("tracking.detector_interval_while_tracking must be >= 1.")
    if config.tracking.reacquire_after_failed_tracker_frames <= 0:
        raise ConfigError("tracking.reacquire_after_failed_tracker_frames must be >= 1.")
    if config.tracking.max_tracker_only_frames_without_detection <= 0:
        raise ConfigError("tracking.max_tracker_only_frames_without_detection must be >= 1.")
    if not 0.0 <= config.tracking.smoothing_alpha <= 1.0:
        raise ConfigError("tracking.smoothing_alpha must be between 0 and 1.")
    if not 0.0 <= config.tracking.acquire_confidence_threshold <= 1.0:
        raise ConfigError("tracking.acquire_confidence_threshold must be between 0 and 1.")
    if not 0.0 <= config.tracking.keep_confidence_threshold <= 1.0:
        raise ConfigError("tracking.keep_confidence_threshold must be between 0 and 1.")
    if config.tracking.keep_confidence_threshold > config.tracking.acquire_confidence_threshold:
        raise ConfigError("tracking.keep_confidence_threshold cannot exceed acquire_confidence_threshold.")
    if config.tracking.confirm_frames <= 0:
        raise ConfigError("tracking.confirm_frames must be >= 1.")
    if config.tracking.reconfirm_frames <= 0:
        raise ConfigError("tracking.reconfirm_frames must be >= 1.")
    if config.tracking.hold_without_detection_frames < 0:
        raise ConfigError("tracking.hold_without_detection_frames must be >= 0.")
    if config.tracking.lost_transition_frames <= 0:
        raise ConfigError("tracking.lost_transition_frames must be >= 1.")
    if config.tracking.hard_remove_frames < 0:
        raise ConfigError("tracking.hard_remove_frames must be >= 0.")
    if not 0.0 <= config.tracking.min_confirm_confidence <= 1.0:
        raise ConfigError("tracking.min_confirm_confidence must be between 0 and 1.")
    if config.tracking.max_missing_frames < 0:
        raise ConfigError("tracking.max_missing_frames must be >= 0.")
    if config.tracking.max_missing_frames < config.tracking.hold_without_detection_frames:
        raise ConfigError("tracking.max_missing_frames must be >= hold_without_detection_frames.")
    if config.tracking.hard_remove_frames < config.tracking.max_missing_frames:
        raise ConfigError("tracking.hard_remove_frames must be >= max_missing_frames.")
    if config.tracking.hard_remove_frames < max(
        config.tracking.max_missing_frames,
        config.tracking.hold_without_detection_frames + config.tracking.lost_transition_frames,
    ):
        raise ConfigError(
            "tracking.hard_remove_frames must be >= "
            "max(max_missing_frames, hold_without_detection_frames + lost_transition_frames)."
        )
    if config.tracking.reacquire_max_frames < 0:
        raise ConfigError("tracking.reacquire_max_frames must be >= 0.")
    if config.tracking.reacquire_max_frames > config.tracking.max_missing_frames:
        raise ConfigError("tracking.reacquire_max_frames cannot exceed max_missing_frames.")
    if not 0.0 <= config.tracking.iou_gate <= 1.0:
        raise ConfigError("tracking.iou_gate must be between 0 and 1.")
    if not 0.0 <= config.tracking.soft_iou_gate <= 1.0:
        raise ConfigError("tracking.soft_iou_gate must be between 0 and 1.")
    if config.tracking.center_distance_gate <= 0:
        raise ConfigError("tracking.center_distance_gate must be positive.")
    if config.tracking.soft_center_distance_gate <= 0:
        raise ConfigError("tracking.soft_center_distance_gate must be positive.")
    if config.tracking.min_area_ratio <= 0:
        raise ConfigError("tracking.min_area_ratio must be positive.")
    if config.tracking.max_area_ratio < config.tracking.min_area_ratio:
        raise ConfigError("tracking.max_area_ratio must be >= min_area_ratio.")
    if config.tracking.soft_min_area_ratio <= 0:
        raise ConfigError("tracking.soft_min_area_ratio must be positive.")
    if config.tracking.soft_max_area_ratio < config.tracking.soft_min_area_ratio:
        raise ConfigError("tracking.soft_max_area_ratio must be >= soft_min_area_ratio.")
    if config.tracking.max_active_tracks <= 0:
        raise ConfigError("tracking.max_active_tracks must be >= 1.")
    if config.tracking.display_sort_mode != "top_to_bottom_left_to_right":
        raise ConfigError("tracking.display_sort_mode must be 'top_to_bottom_left_to_right'.")
    if config.logging.coordinates_log_format not in {"csv", "jsonl"}:
        raise ConfigError("logging.coordinates_log_format must be 'csv' or 'jsonl'.")

    if config.scene_zones.coordinates_mode not in {"normalized", "pixels"}:
        raise ConfigError("scene_zones.coordinates_mode must be 'normalized' or 'pixels'.")
    if not 0.0 <= config.scene_zones.bbox_overlap_threshold <= 1.0:
        raise ConfigError("scene_zones.bbox_overlap_threshold must be between 0 and 1.")
    seen_zone_names: set[str] = set()
    for zone in config.scene_zones.zones:
        if zone.name in seen_zone_names:
            raise ConfigError(f"scene_zones has duplicate zone name '{zone.name}'.")
        seen_zone_names.add(zone.name)

        if zone.shape_type == "rect":
            if None in {zone.x1, zone.y1, zone.x2, zone.y2}:
                raise ConfigError(f"scene zone '{zone.name}' rect requires x1, y1, x2, y2.")
            if zone.x2 <= zone.x1 or zone.y2 <= zone.y1:
                raise ConfigError(
                    f"scene zone '{zone.name}' rect coordinates must satisfy x2>x1 and y2>y1."
                )
            values = (zone.x1, zone.y1, zone.x2, zone.y2)
        else:
            if len(zone.points) < 3:
                raise ConfigError(f"scene zone '{zone.name}' polygon requires at least 3 points.")
            values = tuple(value for point in zone.points for value in point)

        if zone.coordinates_mode == "normalized":
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ConfigError(
                    f"scene zone '{zone.name}' normalized coordinates must be between 0 and 1."
                )
        else:
            if any(value < 0.0 for value in values):
                raise ConfigError(f"scene zone '{zone.name}' pixel coordinates must be >= 0.")

    if config.surface_alert.cooldown_seconds < 0:
        raise ConfigError("surface_alert.cooldown_seconds must be >= 0.")
    if config.surface_alert.alert_point_mode not in {"crosshair_center"}:
        raise ConfigError("surface_alert.alert_point_mode must be 'crosshair_center'.")
    if config.surface_alert.min_interval_per_track < 0:
        raise ConfigError("surface_alert.min_interval_per_track must be >= 0.")
    if config.surface_alert.global_min_interval < 0:
        raise ConfigError("surface_alert.global_min_interval must be >= 0.")
    if config.surface_alert.repeat_interval_seconds < 0:
        raise ConfigError("surface_alert.repeat_interval_seconds must be >= 0.")
    if config.surface_alert.overlay_message_frames < 0:
        raise ConfigError("surface_alert.overlay_message_frames must be >= 0.")
    allowed_zone_types = {zone_type.value for zone_type in ZoneType}
    invalid_continuous_zone_types = [
        zone_type
        for zone_type in config.surface_alert.continuous_zone_types
        if zone_type not in allowed_zone_types
    ]
    if invalid_continuous_zone_types:
        raise ConfigError(
            "surface_alert.continuous_zone_types contains unsupported values: "
            f"{invalid_continuous_zone_types}"
        )
    if config.surface_alert.sound_file:
        sound_path = Path(config.surface_alert.sound_file).expanduser()
        if sound_path.exists() and not sound_path.is_file():
            raise ConfigError("surface_alert.sound_file must point to a file when it exists.")

    if config.alert_recording.prebuffer_seconds < 0:
        raise ConfigError("alert_recording.prebuffer_seconds must be >= 0.")
    if config.alert_recording.postbuffer_seconds < 0:
        raise ConfigError("alert_recording.postbuffer_seconds must be >= 0.")
    if config.alert_recording.fps_fallback <= 0:
        raise ConfigError("alert_recording.fps_fallback must be > 0.")
    if len(config.alert_recording.codec) != 4:
        raise ConfigError("alert_recording.codec must be a four-character FOURCC string.")
    if not config.alert_recording.output_dir.strip():
        raise ConfigError("alert_recording.output_dir must be non-empty.")
    alert_output_path = Path(config.alert_recording.output_dir).expanduser()
    if alert_output_path.exists() and not alert_output_path.is_dir():
        raise ConfigError("alert_recording.output_dir must point to a directory.")

    if config.resize.enabled and (
        not config.resize.width
        or not config.resize.height
        or config.resize.width <= 0
        or config.resize.height <= 0
    ):
        raise ConfigError("resize.width and resize.height must be positive when resize is enabled.")

    if config.output.save_output and not config.output.output_path:
        raise ConfigError("output.output_path is required when save_output=true.")
