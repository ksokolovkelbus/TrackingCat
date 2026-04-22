from __future__ import annotations

import logging
from dataclasses import replace

import cv2
import numpy as np

from app.config import ConfigError, load_config, save_scene_zones_config
from app.models import AppConfig, FrameShape, SceneZone, SceneZonesConfig, ZoneType
from app.overlay import OverlayRenderer
from app.video_source import VideoSource, VideoSourceError
from app.zones import (
    convert_zone_to_coordinates_mode,
    normalize_point,
    pick_zone_at_point,
)


def build_normalized_rect_zone(
    name: str,
    zone_type: ZoneType,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
    frame_shape: FrameShape,
    color: tuple[int, int, int] | None = None,
) -> SceneZone:
    start_x, start_y = normalize_point(start_point, frame_shape)
    end_x, end_y = normalize_point(end_point, frame_shape)
    return SceneZone(
        name=name,
        enabled=True,
        zone_type=zone_type,
        shape_type="rect",
        coordinates_mode="normalized",
        color=color,
        x1=min(start_x, end_x),
        y1=min(start_y, end_y),
        x2=max(start_x, end_x),
        y2=max(start_y, end_y),
    )


def build_normalized_polygon_zone(
    name: str,
    zone_type: ZoneType,
    points: list[tuple[int, int]],
    frame_shape: FrameShape,
    color: tuple[int, int, int] | None = None,
) -> SceneZone:
    normalized_points = tuple(normalize_point(point, frame_shape) for point in points)
    return SceneZone(
        name=name,
        enabled=True,
        zone_type=zone_type,
        shape_type="polygon",
        coordinates_mode="normalized",
        color=color,
        points=normalized_points,
    )


class ZoneEditor:
    def __init__(
        self,
        config: AppConfig,
        config_path: str,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._config_path = config_path
        self._logger = logger
        self._source = VideoSource(config=config.source, logger=logger)
        self._overlay = OverlayRenderer(config.overlay)
        self._window_name = f"{config.output.window_name} | Zone Editor"
        self._scene_zones = SceneZonesConfig(
            enabled=config.scene_zones.enabled,
            draw_zones=True,
            draw_track_locations=config.scene_zones.draw_track_locations,
            surface_priority_over_floor=config.scene_zones.surface_priority_over_floor,
            bbox_overlap_threshold=config.scene_zones.bbox_overlap_threshold,
            coordinates_mode="normalized",
            zone_editor_enabled=True,
            zones=list(config.scene_zones.zones),
        )
        self._selected_zone_name: str | None = self._scene_zones.zones[-1].name if self._scene_zones.zones else None
        self._draft_mode = "rect"
        self._current_zone_type = ZoneType.SURFACE
        self._current_zone_name = self._suggest_zone_name(self._current_zone_type)
        self._draft_points: list[tuple[int, int]] = []
        self._rect_start: tuple[int, int] | None = None
        self._rect_preview_end: tuple[int, int] | None = None
        self._dragging_rect = False
        self._mouse_position: tuple[int, int] = (0, 0)
        self._status_message = "Left mouse: draw/select | Right mouse: finish polygon"
        self._text_input_active = False
        self._text_input_buffer = ""
        self._last_frame: np.ndarray | None = None

    def run(self) -> int:
        try:
            self._source.open()
        except VideoSourceError as exc:
            self._logger.error(str(exc))
            return 3

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

        try:
            while True:
                ok, frame = self._source.read()
                if ok and frame is not None and frame.size > 0:
                    frame = self._maybe_resize_frame(frame)
                    self._last_frame = frame.copy()

                if self._last_frame is None:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord("q"):
                        return 0
                    continue

                canvas = self._last_frame.copy()
                self._overlay.draw_scene_zones(
                    canvas,
                    self._scene_zones.zones,
                    selected_zone_name=self._selected_zone_name,
                )
                self._draw_draft(canvas)
                self._overlay.draw_zone_editor_status(canvas, self._build_status_lines(canvas.shape))

                cv2.imshow(self._window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key, canvas.shape):
                    return 0
        finally:
            self._source.release()
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                self._logger.debug("Zone editor window cleanup failed.", exc_info=True)

    def _handle_key(self, key: int, frame_shape: FrameShape) -> bool:
        if key == 255:
            return False
        if self._text_input_active:
            self._handle_text_input(key)
            return False

        if key == ord("q"):
            return True
        if key == ord("r"):
            self._draft_mode = "rect"
            self._clear_draft()
            self._status_message = "Drawing mode: rect"
            return False
        if key == ord("p"):
            self._draft_mode = "polygon"
            self._clear_draft()
            self._status_message = "Drawing mode: polygon"
            return False
        if key == ord("f"):
            self._current_zone_type = ZoneType.FLOOR
            self._current_zone_name = self._suggest_zone_name(self._current_zone_type)
            self._status_message = "Zone type: floor"
            return False
        if key == ord("s"):
            self._current_zone_type = ZoneType.SURFACE
            self._current_zone_name = self._suggest_zone_name(self._current_zone_type)
            self._status_message = "Zone type: surface"
            return False
        if key == ord("x"):
            self._current_zone_type = ZoneType.RESTRICTED
            self._current_zone_name = self._suggest_zone_name(self._current_zone_type)
            self._status_message = "Zone type: restricted"
            return False
        if key == ord("u"):
            self._undo()
            return False
        if key == ord("c"):
            self._clear_draft()
            self._status_message = "Draft cleared"
            return False
        if key == ord("d"):
            self._delete_selected_zone()
            return False
        if key == ord("n"):
            self._text_input_active = True
            self._text_input_buffer = self._current_zone_name
            self._status_message = "Type zone name and press Enter"
            return False
        if key == ord("w"):
            self._save_zones(frame_shape)
            return False
        if key == ord("l"):
            self._reload_zones()
            return False
        return False

    def _handle_text_input(self, key: int) -> None:
        if key in {13, 10}:
            proposed_name = self._text_input_buffer.strip()
            if proposed_name:
                self._current_zone_name = proposed_name
                self._status_message = f"Zone name set: {proposed_name}"
            self._text_input_active = False
            return
        if key == 27:
            self._text_input_active = False
            self._status_message = "Zone name input cancelled"
            return
        if key in {8, 127}:
            self._text_input_buffer = self._text_input_buffer[:-1]
            return
        if 32 <= key <= 126:
            self._text_input_buffer += chr(key)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        self._mouse_position = (x, y)
        if self._last_frame is None or self._text_input_active:
            return
        frame_shape = self._last_frame.shape

        if event == cv2.EVENT_MOUSEMOVE and self._dragging_rect:
            self._rect_preview_end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._try_select_zone((x, y), frame_shape):
                return
            if self._draft_mode == "rect":
                self._dragging_rect = True
                self._rect_start = (x, y)
                self._rect_preview_end = (x, y)
                return
            self._draft_points.append((x, y))
            self._status_message = f"Polygon points: {len(self._draft_points)}"
            return

        if event == cv2.EVENT_LBUTTONUP and self._dragging_rect:
            self._dragging_rect = False
            self._rect_preview_end = (x, y)
            if self._rect_start is not None and self._rect_preview_end is not None:
                self._finish_rect(frame_shape)
            return

        if event == cv2.EVENT_RBUTTONDOWN and self._draft_mode == "polygon":
            self._finish_polygon(frame_shape)

    def _try_select_zone(self, point: tuple[int, int], frame_shape: FrameShape) -> bool:
        if self._draft_points or self._rect_start is not None:
            return False
        selected_zone = pick_zone_at_point(self._scene_zones.zones, point, frame_shape)
        if selected_zone is None:
            return False
        self._selected_zone_name = selected_zone.name
        self._current_zone_name = selected_zone.name
        self._current_zone_type = selected_zone.zone_type
        self._status_message = f"Selected zone: {selected_zone.name}"
        return True

    def _finish_rect(self, frame_shape: FrameShape) -> None:
        assert self._rect_start is not None and self._rect_preview_end is not None
        if abs(self._rect_start[0] - self._rect_preview_end[0]) < 4 or abs(self._rect_start[1] - self._rect_preview_end[1]) < 4:
            self._status_message = "Rectangle is too small"
            self._clear_draft()
            return
        zone_name = self._ensure_unique_zone_name(self._current_zone_name)
        zone = build_normalized_rect_zone(
            name=zone_name,
            zone_type=self._current_zone_type,
            start_point=self._rect_start,
            end_point=self._rect_preview_end,
            frame_shape=frame_shape,
        )
        self._scene_zones.zones.append(zone)
        self._selected_zone_name = zone.name
        self._status_message = f"Zone created: {zone.name}"
        self._clear_draft()
        self._current_zone_name = self._suggest_zone_name(self._current_zone_type)

    def _finish_polygon(self, frame_shape: FrameShape) -> None:
        if len(self._draft_points) < 3:
            self._status_message = "Polygon needs at least 3 points"
            return
        zone_name = self._ensure_unique_zone_name(self._current_zone_name)
        zone = build_normalized_polygon_zone(
            name=zone_name,
            zone_type=self._current_zone_type,
            points=self._draft_points,
            frame_shape=frame_shape,
        )
        self._scene_zones.zones.append(zone)
        self._selected_zone_name = zone.name
        self._status_message = f"Zone created: {zone.name}"
        self._clear_draft()
        self._current_zone_name = self._suggest_zone_name(self._current_zone_type)

    def _undo(self) -> None:
        if self._draft_mode == "polygon" and self._draft_points:
            self._draft_points.pop()
            self._status_message = "Removed last polygon point"
            return
        if self._draft_mode == "rect" and (self._rect_start is not None or self._dragging_rect):
            self._clear_draft()
            self._status_message = "Rectangle draft cleared"
            return

        if not self._scene_zones.zones:
            self._status_message = "Nothing to undo"
            return

        if self._selected_zone_name is not None:
            for index, zone in enumerate(self._scene_zones.zones):
                if zone.name == self._selected_zone_name:
                    deleted = self._scene_zones.zones.pop(index)
                    self._selected_zone_name = self._scene_zones.zones[-1].name if self._scene_zones.zones else None
                    self._status_message = f"Removed zone: {deleted.name}"
                    return

        deleted = self._scene_zones.zones.pop()
        self._selected_zone_name = self._scene_zones.zones[-1].name if self._scene_zones.zones else None
        self._status_message = f"Removed zone: {deleted.name}"

    def _delete_selected_zone(self) -> None:
        if self._selected_zone_name is None:
            self._status_message = "No selected zone"
            return
        for index, zone in enumerate(self._scene_zones.zones):
            if zone.name == self._selected_zone_name:
                deleted = self._scene_zones.zones.pop(index)
                self._selected_zone_name = self._scene_zones.zones[-1].name if self._scene_zones.zones else None
                self._status_message = f"Deleted zone: {deleted.name}"
                return
        self._status_message = "Selected zone not found"

    def _save_zones(self, frame_shape: FrameShape) -> None:
        normalized_zones = [
            convert_zone_to_coordinates_mode(zone, frame_shape, "normalized")
            for zone in self._scene_zones.zones
        ]
        payload = replace(
            self._scene_zones,
            coordinates_mode="normalized",
            zone_editor_enabled=False,
            zones=normalized_zones,
        )
        try:
            save_scene_zones_config(self._config_path, payload)
        except (ConfigError, OSError) as exc:
            self._status_message = f"Save failed: {exc}"
            self._logger.error("Zone editor save failed: %s", exc)
            return
        self._scene_zones = payload
        self._status_message = f"Zones saved to {self._config_path}"
        self._logger.info("Zone editor saved %d zones to %s", len(payload.zones), self._config_path)

    def _reload_zones(self) -> None:
        try:
            config = load_config(self._config_path)
        except ConfigError as exc:
            self._status_message = f"Reload failed: {exc}"
            self._logger.error("Zone editor reload failed: %s", exc)
            return

        self._scene_zones = SceneZonesConfig(
            enabled=config.scene_zones.enabled,
            draw_zones=True,
            draw_track_locations=config.scene_zones.draw_track_locations,
            surface_priority_over_floor=config.scene_zones.surface_priority_over_floor,
            bbox_overlap_threshold=config.scene_zones.bbox_overlap_threshold,
            coordinates_mode=config.scene_zones.coordinates_mode,
            zone_editor_enabled=True,
            zones=list(config.scene_zones.zones),
        )
        self._selected_zone_name = self._scene_zones.zones[-1].name if self._scene_zones.zones else None
        self._status_message = f"Reloaded {len(self._scene_zones.zones)} zones from YAML"

    def _clear_draft(self) -> None:
        self._draft_points.clear()
        self._rect_start = None
        self._rect_preview_end = None
        self._dragging_rect = False

    def _draw_draft(self, frame: np.ndarray) -> None:
        color = self._draft_color()
        if self._draft_mode == "rect" and self._rect_start is not None and self._rect_preview_end is not None:
            cv2.rectangle(frame, self._rect_start, self._rect_preview_end, color, 2)
            return

        if self._draft_mode == "polygon" and self._draft_points:
            points = np.array(self._draft_points, dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
            for point in self._draft_points:
                cv2.circle(frame, point, 3, color, thickness=-1)
            if self._mouse_position and len(self._draft_points) >= 1:
                cv2.line(frame, self._draft_points[-1], self._mouse_position, color, 1)

    def _build_status_lines(self, frame_shape: FrameShape) -> list[str]:
        normalized_mouse = normalize_point(self._mouse_position, frame_shape)
        zone_names = ", ".join(
            f"[{zone.name}]" if zone.name == self._selected_zone_name else zone.name
            for zone in self._scene_zones.zones[-6:]
        ) or "-"
        name_line = self._text_input_buffer if self._text_input_active else self._current_zone_name
        return [
            "ZONE EDITOR",
            f"mode: {self._draft_mode}",
            f"zone type: {self._current_zone_type.value}",
            f"name: {name_line}",
            f"mouse: {self._mouse_position[0]},{self._mouse_position[1]} | "
            f"normalized: {normalized_mouse[0]:.3f},{normalized_mouse[1]:.3f}",
            f"selected: {self._selected_zone_name or '-'}",
            f"zones: {zone_names}",
            "keys: r rect | p poly | f floor | s surface | x restricted",
            "keys: n name | u undo | c clear | d delete | w save | l load | q quit",
            self._status_message,
        ]

    def _draft_color(self) -> tuple[int, int, int]:
        if self._current_zone_type == ZoneType.FLOOR:
            return self._config.overlay.accent_color
        if self._current_zone_type == ZoneType.RESTRICTED:
            return self._config.overlay.error_color
        return self._config.overlay.warning_color

    def _ensure_unique_zone_name(self, base_name: str) -> str:
        proposed_name = base_name.strip() or self._suggest_zone_name(self._current_zone_type)
        existing_names = {zone.name for zone in self._scene_zones.zones}
        if proposed_name not in existing_names:
            return proposed_name
        suffix = 2
        while f"{proposed_name}_{suffix}" in existing_names:
            suffix += 1
        return f"{proposed_name}_{suffix}"

    def _suggest_zone_name(self, zone_type: ZoneType) -> str:
        prefix = {
            ZoneType.FLOOR: "floor",
            ZoneType.SURFACE: "surface",
            ZoneType.RESTRICTED: "restricted",
        }[zone_type]
        matching_names = [zone.name for zone in self._scene_zones.zones if zone.zone_type == zone_type]
        return f"{prefix}_{len(matching_names) + 1}"

    def _maybe_resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._config.resize.enabled:
            return frame
        return cv2.resize(
            frame,
            (self._config.resize.width, self._config.resize.height),
            interpolation=cv2.INTER_LINEAR,
        )
