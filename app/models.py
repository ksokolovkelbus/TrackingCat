from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, TypeAlias


SourceType: TypeAlias = Literal["webcam", "file", "rtsp", "http_snapshot"]
TargetSelectionStrategy: TypeAlias = Literal[
    "largest_area",
    "highest_confidence",
    "closest_to_center",
]
CoordinateLogFormat: TypeAlias = Literal["csv", "jsonl"]
DisplaySortMode: TypeAlias = Literal["top_to_bottom_left_to_right"]
ZoneCoordinatesMode: TypeAlias = Literal["normalized", "pixels"]
ZoneShapeType: TypeAlias = Literal["rect", "polygon"]
AlertPointMode: TypeAlias = Literal["crosshair_center"]
SurfaceEventType: TypeAlias = Literal["entered_surface", "returned_to_floor"]
Color: TypeAlias = tuple[int, int, int]
BBox: TypeAlias = tuple[int, int, int, int]
FloatBBox: TypeAlias = tuple[float, float, float, float]
FrameShape: TypeAlias = tuple[int, int] | tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> FloatBBox:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class DetectionOverlay:
    display_number: int | None
    square_bbox: BBox
    center_x: float
    center_y: float
    normalized_x: float
    normalized_y: float
    confidence: float
    class_name: str = "cat"

    @property
    def center(self) -> tuple[int, int]:
        return int(round(self.center_x)), int(round(self.center_y))


class TrackState(StrEnum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    HELD = "HELD"
    LOST = "LOST"
    REMOVED = "REMOVED"


class TrackingPipelineState(StrEnum):
    SEARCH = "SEARCH"
    TRACK_ONLY = "TRACK_ONLY"
    REACQUIRE = "REACQUIRE"
    LOST = "LOST"


class ZoneType(StrEnum):
    FLOOR = "floor"
    SURFACE = "surface"
    RESTRICTED = "restricted"


@dataclass(frozen=True, slots=True)
class SceneZone:
    name: str
    enabled: bool
    zone_type: ZoneType
    shape_type: ZoneShapeType
    coordinates_mode: ZoneCoordinatesMode = "normalized"
    color: Color | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    points: tuple[tuple[float, float], ...] = ()


@dataclass(slots=True)
class TrackLocationState:
    track_id: int
    current_zone_name: str | None = None
    current_zone_type: ZoneType | None = None
    previous_zone_name: str | None = None
    previous_zone_type: ZoneType | None = None
    is_on_surface: bool = False
    last_surface_event_ts: float = 0.0
    last_sound_ts: float = 0.0
    last_zone_change_ts: float = 0.0
    last_frame_index: int = 0
    last_alert_zone_name: str | None = None
    alert_entered_ts: float = 0.0
    floor_frames: int = 0
    previous_floor_frames: int = 0
    current_zone_frames: int = 0

    @property
    def location_label(self) -> str:
        if self.current_zone_type is None:
            return "unknown"
        if self.current_zone_type == ZoneType.FLOOR:
            return (
                f"floor/{self.current_zone_name}"
                if self.current_zone_name
                else "floor"
            )
        if self.current_zone_type == ZoneType.RESTRICTED:
            return (
                f"restricted/{self.current_zone_name}"
                if self.current_zone_name
                else "restricted"
            )
        return (
            f"surface/{self.current_zone_name}"
            if self.current_zone_name
            else "surface"
        )


@dataclass(frozen=True, slots=True)
class SurfaceEvent:
    event_type: SurfaceEventType
    track_id: int
    zone_name: str
    zone_type: ZoneType
    frame_index: int
    timestamp: float
    message: str


@dataclass(frozen=True, slots=True)
class ActiveAlertTrack:
    track_id: int
    display_number: int | None
    zone_name: str
    zone_type: ZoneType
    entered_at_ts: float
    dwell_seconds: float


@dataclass(frozen=True, slots=True)
class AlertRecordingState:
    output_path: str
    started_at_ts: float
    started_at_iso: str
    current_wallclock_iso: str
    zone_names: tuple[str, ...]
    active_track_ids: tuple[int, ...]
    active_display_numbers: tuple[int, ...]
    incident_track_ids: tuple[int, ...] = ()
    incident_display_numbers: tuple[int, ...] = ()
    elapsed_seconds: float = 0.0
    frame_count: int = 0
    is_recording: bool = True
    postbuffer_active: bool = False


@dataclass(slots=True)
class AlertIncident:
    video_path: str
    metadata_path: str
    started_at_ts: float
    ended_at_ts: float
    started_at_iso: str
    ended_at_iso: str
    zone_names: tuple[str, ...]
    track_ids: tuple[int, ...]
    display_numbers: tuple[int, ...]
    duration_seconds: float
    total_frames: int
    dwell_seconds_by_track_id: dict[int, float] = field(default_factory=dict)
    first_entered_at_by_track_id: dict[int, float] = field(default_factory=dict)
    last_exited_at_by_track_id: dict[int, float] = field(default_factory=dict)
    zone_names_by_track_id: dict[int, tuple[str, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class Track:
    track_id: int
    display_number: int | None
    state: TrackState
    bbox: FloatBBox
    square_bbox: BBox
    center_x: float
    center_y: float
    normalized_x: float
    normalized_y: float
    confidence: float
    age: int
    hits: int
    misses: int
    consecutive_hits: int
    consecutive_misses: int
    first_seen_frame: int
    last_seen_frame: int
    last_detection_frame: int = 0
    last_update_ts: float = 0.0
    reconfirm_hits: int = 0
    held_frames: int = 0
    tracker_only_updates: int = 0
    tracker_failures: int = 0
    class_name: str = "cat"
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    predicted_bbox: FloatBBox | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return int(round(self.center_x)), int(round(self.center_y))

    @property
    def is_active(self) -> bool:
        return self.state in {
            TrackState.TENTATIVE,
            TrackState.CONFIRMED,
            TrackState.HELD,
            TrackState.LOST,
        }

    def is_displayed(self, frame_index: int) -> bool:
        if self.state != TrackState.CONFIRMED:
            return False
        return self.last_detection_frame == frame_index




@dataclass(slots=True)
class StageTimings:
    source_read_ms: float = 0.0
    resize_ms: float = 0.0
    detect_or_track_ms: float = 0.0
    surface_monitor_ms: float = 0.0
    overlay_ms: float = 0.0
    alert_recording_ms: float = 0.0
    output_write_ms: float = 0.0
    window_ms: float = 0.0
    total_frame_ms: float = 0.0

@dataclass(slots=True)
class FrameTrackingSummary:
    frame_index: int
    frame_width: int
    frame_height: int
    tracking_enabled: bool
    detections_count: int
    pipeline_state: TrackingPipelineState = TrackingPipelineState.SEARCH
    yolo_ran_this_frame: bool = False
    tracker_updates_count: int = 0
    tracker_failures_count: int = 0
    raw_detections_count: int = 0
    cat_detections_count: int = 0
    acquire_candidate_count: int = 0
    keep_candidate_count: int = 0
    detection_overlays: list[DetectionOverlay] = field(default_factory=list)
    visible_tracks: list[Track] = field(default_factory=list)
    held_tracks: list[Track] = field(default_factory=list)
    tentative_tracks: list[Track] = field(default_factory=list)
    lost_tracks: list[Track] = field(default_factory=list)
    removed_track_ids: list[int] = field(default_factory=list)
    track_location_states: dict[int, TrackLocationState] = field(default_factory=dict)
    surface_events: list[SurfaceEvent] = field(default_factory=list)
    suppressed_surface_events: list[SurfaceEvent] = field(default_factory=list)
    active_alert_tracks: list[ActiveAlertTrack] = field(default_factory=list)
    surface_overlay_message: str | None = None
    alert_recording_state: AlertRecordingState | None = None
    stage_timings: StageTimings = field(default_factory=StageTimings)

    @property
    def visible_count(self) -> int:
        return len(self.visible_tracks)

    @property
    def detection_overlay_count(self) -> int:
        return len(self.detection_overlays)

    @property
    def held_count(self) -> int:
        return len(self.held_tracks)

    @property
    def confirmed_count(self) -> int:
        return self.visible_count - self.held_count

    @property
    def tentative_count(self) -> int:
        return len(self.tentative_tracks)

    @property
    def lost_count(self) -> int:
        return len(self.lost_tracks)

    @property
    def active_tracks_count(self) -> int:
        return self.visible_count + self.tentative_count + self.lost_count


@dataclass(slots=True)
class Target:
    detection: Detection
    square_bbox: BBox
    center_x: float
    center_y: float
    normalized_x: float
    normalized_y: float
    display_id: int | None = None
    is_predicted: bool = False
    lost_frames: int = 0

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def center(self) -> tuple[int, int]:
        return int(round(self.center_x)), int(round(self.center_y))


@dataclass(slots=True)
class OverlayConfig:
    minimal_overlay: bool = False
    debug_overlay: bool = True
    show_fps: bool = True
    show_debug_counters: bool = True
    show_track_boxes: bool = False
    show_track_crosshair: bool = True
    show_track_labels: bool = True
    show_detection_labels: bool = True
    show_zone_labels: bool = True
    show_cat_count: bool = False
    box_color: Color = (0, 200, 255)
    predicted_box_color: Color = (0, 140, 255)
    secondary_box_color: Color = (150, 150, 150)
    marker_color: Color = (0, 0, 255)
    marker_crosshair_color: Color = (255, 255, 255)
    marker_dot_color: Color = (0, 0, 255)
    text_color: Color = (240, 240, 240)
    accent_color: Color = (80, 220, 120)
    warning_color: Color = (0, 180, 255)
    error_color: Color = (0, 80, 255)
    background_color: Color = (16, 16, 16)
    box_thickness: int = 2
    secondary_box_thickness: int = 1
    marker_thickness: int = 2
    marker_size: int = 18
    marker_gap: int = 6
    center_radius: int = 3
    marker_ring_radius: int = 12
    marker_dot_radius: int = 3
    font_scale: float = 0.55
    font_thickness: int = 1
    padding: int = 8
    line_height: int = 22
    show_normalized_coords: bool = True


@dataclass(slots=True)
class SourceConfig:
    source_type: SourceType = "webcam"
    source_path: str | None = None
    camera_index: int = 0
    camera_width: int | None = 1280
    camera_height: int | None = 720
    process_every_n_frames: int = 1
    stream_url: str | None = None
    reconnect_delay_seconds: float = 1.0
    max_reconnect_attempts: int = 5
    read_fail_threshold: int = 3
    buffer_size: int = 1
    snapshot_timeout_seconds: float = 2.0
    snapshot_use_cache_bust: bool = True

    def resolved_source(self) -> int | str:
        if self.source_type == "webcam":
            return self.camera_index
        if self.source_type == "file":
            if not self.source_path:
                raise ValueError("source_path is required when source_type='file'.")
            return self.source_path
        if not self.stream_url:
            raise ValueError(f"stream_url is required when source_type='{self.source_type}'.")
        return self.stream_url


@dataclass(slots=True)
class DetectorConfig:
    model_path: str = "yolo26s.pt"
    imgsz: int = 640
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.45
    device: str = "cpu"
    class_name: str = "cat"
    max_frame_area_ratio: float = 1.0


@dataclass(slots=True)
class TrackingConfig:
    tracking_enabled: bool = True
    multi_target_enabled: bool = True
    tracker_only_mode_after_confirm: bool = True
    detector_interval_while_tracking: int = 6
    reacquire_after_failed_tracker_frames: int = 2
    max_tracker_only_frames_without_detection: int = 10
    acquire_confidence_threshold: float = 0.25
    keep_confidence_threshold: float = 0.08
    confirm_frames: int = 2
    reconfirm_frames: int = 2
    hold_without_detection_frames: int = 8
    lost_transition_frames: int = 4
    max_missing_frames: int = 12
    hard_remove_frames: int = 18
    min_confirm_confidence: float = 0.25
    reacquire_max_frames: int = 6
    iou_gate: float = 0.20
    center_distance_gate: float = 140.0
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.8
    soft_iou_gate: float = 0.08
    soft_center_distance_gate: float = 180.0
    soft_min_area_ratio: float = 0.25
    soft_max_area_ratio: float = 3.2
    smoothing_alpha: float = 0.30
    preserve_confirmed_tracks: bool = True
    never_drop_confirmed_on_single_bad_frame: bool = True
    keep_lost_tracks: bool = True
    max_active_tracks: int = 16
    use_motion_prediction: bool = True
    display_sort_mode: DisplaySortMode = "top_to_bottom_left_to_right"
    frame_tracker_backend: FrameTrackerBackend = "auto"


@dataclass(slots=True)
class SceneZonesConfig:
    enabled: bool = True
    draw_zones: bool = True
    draw_track_locations: bool = True
    surface_priority_over_floor: bool = True
    bbox_overlap_threshold: float = 0.15
    coordinates_mode: ZoneCoordinatesMode = "normalized"
    zone_editor_enabled: bool = False
    zones: list[SceneZone] = field(default_factory=list)


@dataclass(slots=True)
class SurfaceAlertConfig:
    enabled: bool = True
    trigger_on_surface_entry: bool = True
    alert_point_mode: AlertPointMode = "crosshair_center"
    trigger_only_from_floor: bool = True
    trigger_from_unknown: bool = False
    min_floor_frames_before_alert: int = 0
    min_zone_frames_before_alert: int = 1
    cooldown_seconds: float = 2.0
    min_interval_per_track: float = 5.0
    global_min_interval: float = 1.0
    repeat_on_same_surface: bool = False
    sound_file: str | None = None
    beep_fallback: bool = True

    # Новый режим: звук играет постоянно, пока есть кот в нужной зоне
    continuous_while_in_zone: bool = True

    # Для каких типов зон держать непрерывный звук
    # Обычно достаточно restricted, чтобы не орало на обычных surface
    continuous_zone_types: tuple[str, ...] = ("restricted",)

    # Интервал между повторами звука в continuous режиме
    # Если используешь свой wav, поставь примерно длину файла
    repeat_interval_seconds: float = 1.0

    # Останавливать звук, когда в нужных зонах никого не осталось
    stop_when_zone_empty: bool = True

    show_overlay_message: bool = True
    overlay_message_frames: int = 20


@dataclass(slots=True)
class AlertRecordingConfig:
    enabled: bool = True
    output_dir: str = "/home/kelbus/PycharmProjects/TrackingCat/alert_video"
    prebuffer_seconds: float = 3.0
    postbuffer_seconds: float = 1.5
    fps_fallback: float = 15.0
    codec: str = "mp4v"
    draw_recording_overlay: bool = True
    include_wallclock_timestamp: bool = True
    include_zone_name: bool = True
    include_track_ids: bool = True
    include_elapsed_seconds: bool = True


@dataclass(slots=True)
class ResizeConfig:
    enabled: bool = False
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class PanTiltControlConfig:
    enabled: bool = False
    manual_control_only: bool = True
    base_url: str = "http://trackingcat-pantilt.local"
    request_timeout_seconds: float = 0.8
    status_poll_interval_seconds: float = 2.0
    default_step_degrees: int = 3
    coarse_step_degrees: int = 8
    show_controls: bool = True
    button_up_direction: str = "left"
    button_down_direction: str = "right"
    button_left_direction: str = "down"
    button_right_direction: str = "up"
    hold_repeat_interval_seconds: float = 0.12
    default_speed_mode: str = "medium"


@dataclass(slots=True)
class OutputConfig:
    show_window: bool = True
    save_output: bool = False
    output_path: str = "output/tracking_cat.mp4"
    window_name: str = "TrackingCat"


@dataclass(slots=True)
class LoggingConfig:
    log_level: str = "INFO"
    per_frame_debug: bool = False
    log_coordinates: bool = True
    coordinates_log_path: str | None = None
    coordinates_log_format: CoordinateLogFormat = "csv"


@dataclass(slots=True)
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    target_selection_strategy: TargetSelectionStrategy = "closest_to_center"
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    pan_tilt: PanTiltControlConfig = field(default_factory=PanTiltControlConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    scene_zones: SceneZonesConfig = field(default_factory=SceneZonesConfig)
    surface_alert: SurfaceAlertConfig = field(default_factory=SurfaceAlertConfig)
    alert_recording: AlertRecordingConfig = field(default_factory=AlertRecordingConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)
