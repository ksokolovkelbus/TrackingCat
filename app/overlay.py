from __future__ import annotations

import cv2
import numpy as np

from app.models import (
    AlertRecordingState,
    DetectionOverlay,
    FrameTrackingSummary,
    OverlayConfig,
    SceneZone,
    Track,
    TrackLocationState,
    TrackState,
    ZoneType,
)
from app.utils import safe_float_text
from app.zones import project_zone


class OverlayRenderer:
    def __init__(self, config: OverlayConfig) -> None:
        self._config = config
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_square_box(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int] | None = None,
        thickness: int | None = None,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color or self._config.box_color,
            thickness or self._config.box_thickness,
        )
        return frame

    def draw_center_marker(self, frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        cx, cy = center
        marker_size = self._config.marker_size
        gap = self._config.marker_gap
        color = self._config.marker_color
        thickness = self._config.marker_thickness

        cv2.line(frame, (cx - marker_size, cy), (cx - gap, cy), color, thickness)
        cv2.line(frame, (cx + gap, cy), (cx + marker_size, cy), color, thickness)
        cv2.line(frame, (cx, cy - marker_size), (cx, cy - gap), color, thickness)
        cv2.line(frame, (cx, cy + gap), (cx, cy + marker_size), color, thickness)
        cv2.circle(frame, (cx, cy), self._config.center_radius, color, thickness=-1)
        return frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        track_location_states: dict[int, TrackLocationState] | None = None,
    ) -> np.ndarray:
        for track in tracks:
            self.draw_track(
                frame,
                track,
                track_location_state=(
                    track_location_states.get(track.track_id)
                    if track_location_states is not None
                    else None
                ),
            )
        return frame

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[DetectionOverlay],
    ) -> np.ndarray:
        for detection in detections:
            self.draw_detection(frame, detection)
        return frame

    def draw_detection(self, frame: np.ndarray, detection: DetectionOverlay) -> np.ndarray:
        self.draw_square_box(frame, detection.square_bbox, color=self._config.secondary_box_color)
        self.draw_center_marker(frame, detection.center)
        if self._config.show_detection_labels:
            x1, y1, _, _ = detection.square_bbox
            lines = self._build_detection_lines(detection)
            self._draw_text_block(frame, lines, origin=(x1, max(y1 - 8, 18)))
        return frame

    def draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        track_location_state: TrackLocationState | None = None,
    ) -> np.ndarray:
        box_color = (
            self._config.predicted_box_color
            if track.state == TrackState.HELD
            else self._config.box_color
        )
        if self._config.show_track_boxes:
            self.draw_square_box(frame, track.square_bbox, color=box_color)
        if self._config.show_track_crosshair:
            self.draw_center_marker(frame, track.center)

        if self._config.show_track_labels:
            x1, y1, _, _ = track.square_bbox
            label_origin = (
                x1,
                max(y1 - 8, 18),
            )
            if not self._config.show_track_boxes:
                label_origin = (
                    min(frame.shape[1] - 10, track.center[0] + 10),
                    max(18, track.center[1] - 10),
                )
            status_text = "held" if track.state == TrackState.HELD else "tracked"
            lines = self._build_track_lines(track, status_text, track_location_state)
            self._draw_text_block(frame, lines, origin=label_origin)
        return frame

    def draw_scene_zones(
        self,
        frame: np.ndarray,
        zones: list[SceneZone],
        selected_zone_name: str | None = None,
    ) -> np.ndarray:
        for zone in zones:
            projected = project_zone(zone, frame.shape)
            color = zone.color or self._zone_color(zone.zone_type)
            thickness = 3 if zone.name == selected_zone_name else 2
            if zone.shape_type == "rect":
                left, top, right, bottom = projected.bounds
                cv2.rectangle(
                    frame,
                    (int(round(left)), int(round(top))),
                    (int(round(right)), int(round(bottom))),
                    color,
                    thickness,
                )
                label_origin = (int(round(left)) + 4, max(18, int(round(top)) - 6))
            else:
                points = np.array(projected.points, dtype=np.int32)
                if len(points) < 3:
                    continue
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)
                centroid_x = int(sum(point[0] for point in projected.points) / len(projected.points))
                centroid_y = int(sum(point[1] for point in projected.points) / len(projected.points))
                label_origin = (centroid_x, max(18, centroid_y))
            if self._config.show_zone_labels:
                self.draw_label(
                    frame,
                    text=f"{zone.zone_type.value}: {zone.name}",
                    origin=label_origin,
                    text_color=color,
                )
        return frame

    def draw_zone_editor_status(self, frame: np.ndarray, lines: list[str]) -> np.ndarray:
        return self._draw_text_block(frame, lines, origin=(12, 24))

    def draw_surface_alert_message(self, frame: np.ndarray, message: str) -> np.ndarray:
        return self.draw_label(
            frame,
            text=message,
            origin=(12, max(32, frame.shape[0] - 44)),
            text_color=self._config.warning_color,
        )

    def draw_alert_recording_status(
        self,
        frame: np.ndarray,
        recording_state: AlertRecordingState,
    ) -> np.ndarray:
        lines = ["ALERT RECORDING"]
        if recording_state.current_wallclock_iso:
            lines.append(f"now: {recording_state.current_wallclock_iso}")
        if recording_state.zone_names:
            lines.append(f"zone: {', '.join(recording_state.zone_names)}")
        if recording_state.active_display_numbers:
            cats = ",".join(str(number) for number in recording_state.active_display_numbers)
            lines.append(f"cats: {cats}")
        elif recording_state.active_track_ids:
            cats = ",".join(str(track_id) for track_id in recording_state.active_track_ids)
            lines.append(f"cats: {cats}")
        else:
            lines.append("cats: -")
        lines.append(f"started: {recording_state.started_at_iso}")
        lines.append(f"elapsed: {self._format_elapsed(recording_state.elapsed_seconds)}")
        if recording_state.postbuffer_active:
            lines.append("state: postbuffer")

        return self._draw_text_block_top_right(
            frame=frame,
            lines=lines,
            first_line_color=self._config.error_color,
        )

    def draw_status(
        self,
        frame: np.ndarray,
        status_text: str,
        summary: FrameTrackingSummary,
        source_status: str,
        mode_text: str,
    ) -> np.ndarray:
        if not self._config.debug_overlay:
            return frame

        detection_line = (
            f"Rendered detections: {summary.detection_overlay_count}"
            if not summary.tracking_enabled
            else f"Tracker inputs: {summary.detections_count}"
        )
        lines = [
            status_text,
            f"Source: {source_status}",
            f"Mode: {mode_text}",
        ]
        if self._config.show_debug_counters:
            lines.extend(
                [
                    f"Tracking enabled: {summary.tracking_enabled}",
                    f"Pipeline: {summary.pipeline_state.value}",
                    f"YOLO ran: {summary.yolo_ran_this_frame}",
                    f"Raw detections: {summary.raw_detections_count}",
                    f"Cat detections: {summary.cat_detections_count}",
                    f"After acquire: {summary.acquire_candidate_count}",
                    f"After keep: {summary.keep_candidate_count}",
                    detection_line,
                    f"Tracker updates: {summary.tracker_updates_count}",
                    f"Tracker failures: {summary.tracker_failures_count}",
                    f"Active tracks: {summary.active_tracks_count}",
                    f"Visible: {summary.visible_count}",
                    f"Confirmed: {summary.confirmed_count}",
                    f"Held: {summary.held_count}",
                    f"Tentative: {summary.tentative_count}",
                    f"Lost: {summary.lost_count}",
                ],
            )
        return self._draw_text_block(frame, lines, origin=(12, 24))

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        if not self._config.show_fps:
            return frame
        return self.draw_label(
            frame,
            text=f"FPS: {safe_float_text(fps, precision=1, default='0.0')}",
            origin=(12, frame.shape[0] - 16),
            text_color=self._config.accent_color,
        )

    def draw_cat_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        if not self._config.show_cat_count:
            return frame
        return self.draw_label(
            frame,
            text=f"Cats: {count}",
            origin=(12, 24),
            text_color=self._config.accent_color,
        )

    def _build_detection_lines(self, detection: DetectionOverlay) -> list[str]:
        if self._config.minimal_overlay:
            return [f"Cat {detection.display_number} | conf {detection.confidence:.2f}"]

        lines = [
            f"Cat {detection.display_number}",
            "status: detected",
            f"conf: {detection.confidence:.3f}",
            f"X: {int(round(detection.center_x))}, Y: {int(round(detection.center_y))}",
        ]
        if self._config.show_normalized_coords:
            lines.append(
                f"Xn: {safe_float_text(detection.normalized_x)}, "
                f"Yn: {safe_float_text(detection.normalized_y)}"
            )
        return lines

    def _build_track_lines(
        self,
        track: Track,
        status_text: str,
        track_location_state: TrackLocationState | None,
    ) -> list[str]:
        if self._config.minimal_overlay:
            if track_location_state is not None:
                return [
                    f"Cat {track.display_number} | {status_text} | {track_location_state.location_label}",
                ]
            return [f"Cat {track.display_number} | {status_text} | conf {track.confidence:.2f}"]

        lines = [
            f"Cat {track.display_number}",
            f"status: {status_text}",
            f"conf: {track.confidence:.3f}",
            f"X: {int(round(track.center_x))}, Y: {int(round(track.center_y))}",
        ]
        if track_location_state is not None:
            lines.append(f"zone: {track_location_state.location_label}")
        if self._config.show_normalized_coords:
            lines.append(
                f"Xn: {safe_float_text(track.normalized_x)}, "
                f"Yn: {safe_float_text(track.normalized_y)}"
            )
        return lines

    def draw_label(
            self,
            frame: np.ndarray,
            text: str,
            origin: tuple[int, int],
            text_color: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        x, y = origin
        cv2.putText(
            frame,
            text,
            (x, y),
            self._font,
            self._config.font_scale,
            text_color or self._config.text_color,
            self._config.font_thickness,
            lineType=cv2.LINE_AA,
        )
        return frame

    def _draw_text_block_top_right(
        self,
        frame: np.ndarray,
        lines: list[str],
        first_line_color: tuple[int, int, int],
    ) -> np.ndarray:
        if not lines:
            return frame

        widths = [
            cv2.getTextSize(
                line,
                self._font,
                self._config.font_scale,
                self._config.font_thickness,
            )[0][0]
            for line in lines
        ]
        max_width = max(widths)
        x = max(12, frame.shape[1] - max_width - 24)
        y = 24

        for index, line in enumerate(lines):
            line_y = y + (index * self._config.line_height)
            if line_y >= frame.shape[0] - 5:
                break
            color = first_line_color if index == 0 else self._config.text_color
            cv2.putText(
                frame,
                line,
                (x, line_y),
                self._font,
                self._config.font_scale,
                color,
                self._config.font_thickness,
                lineType=cv2.LINE_AA,
            )
        return frame

    def _zone_color(self, zone_type: ZoneType) -> tuple[int, int, int]:
        if zone_type == ZoneType.FLOOR:
            return self._config.accent_color
        if zone_type == ZoneType.RESTRICTED:
            return self._config.error_color
        return self._config.warning_color

    def _draw_text_block(
            self,
            frame: np.ndarray,
            lines: list[str],
            origin: tuple[int, int],
    ) -> np.ndarray:
        x, y = origin
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Чуть сдвигаем текст внутрь кадра, чтобы не выходил за границы
        x = max(0, min(x, frame_width - 10))
        y = max(20, min(y, frame_height - 10))

        for index, line in enumerate(lines):
            line_y = y + (index * self._config.line_height)
            if line_y >= frame_height - 5:
                break

            cv2.putText(
                frame,
                line,
                (x, line_y),
                self._font,
                self._config.font_scale,
                self._config.text_color,
                self._config.font_thickness,
                lineType=cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _format_elapsed(elapsed_seconds: float) -> str:
        total_seconds = max(0, int(round(elapsed_seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
