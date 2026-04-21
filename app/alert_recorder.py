from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.models import (
    ActiveAlertTrack,
    AlertIncident,
    AlertRecordingConfig,
    AlertRecordingState,
)
from app.overlay import OverlayRenderer


@dataclass(slots=True)
class _BufferedFrame:
    timestamp: float
    frame: np.ndarray


@dataclass(slots=True)
class _IncidentRuntime:
    writer: cv2.VideoWriter
    temp_output_path: Path
    started_at_ts: float
    started_at_iso: str
    fps: float
    frame_width: int
    frame_height: int
    frame_count: int = 0
    last_written_ts: float = 0.0
    postbuffer_until_ts: float | None = None
    zone_names_seen: set[str] = field(default_factory=set)
    track_ids_seen: set[int] = field(default_factory=set)
    display_numbers_seen: set[int] = field(default_factory=set)
    active_zone_names: tuple[str, ...] = ()
    track_entered_at: dict[int, float] = field(default_factory=dict)
    track_first_entered_at: dict[int, float] = field(default_factory=dict)
    track_last_exited_at: dict[int, float] = field(default_factory=dict)
    track_dwell_seconds: dict[int, float] = field(default_factory=dict)
    track_display_numbers: dict[int, int | None] = field(default_factory=dict)
    track_zone_names: dict[int, set[str]] = field(default_factory=dict)


class AlertRecorder:
    def __init__(
        self,
        config: AlertRecordingConfig,
        logger: logging.Logger,
        overlay_renderer: OverlayRenderer | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._overlay_renderer = overlay_renderer
        self._prebuffer: deque[_BufferedFrame] = deque()
        self._runtime: _IncidentRuntime | None = None
        self._current_state: AlertRecordingState | None = None
        self._last_incident: AlertIncident | None = None

    @property
    def current_state(self) -> AlertRecordingState | None:
        return self._current_state

    @property
    def last_incident(self) -> AlertIncident | None:
        return self._last_incident

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        fps: float,
        active_alert_tracks: list[ActiveAlertTrack],
    ) -> AlertRecordingState | None:
        del frame_index

        if not self._config.enabled:
            return None

        self._prune_prebuffer(timestamp)

        if self._runtime is None:
            if not active_alert_tracks:
                self._append_prebuffer(frame, timestamp)
                self._current_state = None
                return None
            if not self._start_incident(frame=frame, timestamp=timestamp, fps=fps, active_alert_tracks=active_alert_tracks):
                self._append_prebuffer(frame, timestamp)
                self._current_state = None
                return None
            self._flush_prebuffer()
        else:
            self._update_runtime_membership(active_alert_tracks, timestamp)

        state = self._build_recording_state(timestamp, active_alert_tracks)
        self._current_state = state
        if (
            state is not None
            and self._config.draw_recording_overlay
            and self._overlay_renderer is not None
        ):
            self._overlay_renderer.draw_alert_recording_status(frame, state)

        self._write_frame(frame, timestamp)
        self._append_prebuffer(frame, timestamp)

        runtime = self._runtime
        if runtime is not None and not active_alert_tracks:
            if runtime.postbuffer_until_ts is None:
                runtime.postbuffer_until_ts = timestamp + self._config.postbuffer_seconds
            if timestamp >= runtime.postbuffer_until_ts:
                self._finalize_incident(timestamp)
                self._current_state = None
                return None

        return self._current_state

    def close(self) -> AlertIncident | None:
        if self._runtime is not None:
            finalize_ts = self._runtime.last_written_ts or time.time()
            self._finalize_incident(finalize_ts)
        return self._last_incident

    def _start_incident(
        self,
        frame: np.ndarray,
        timestamp: float,
        fps: float,
        active_alert_tracks: list[ActiveAlertTrack],
    ) -> bool:
        output_dir = Path(self._config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        started_at_iso = self._format_wallclock(timestamp)
        timestamp_slug = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
        temp_output_path = self._resolve_unique_path(output_dir / f"{timestamp_slug}_pending.mp4")
        writer = self._create_video_writer(
            output_path=temp_output_path,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            fps=fps,
        )
        if writer is None:
            self._logger.error(
                "alert_recording_failed reason=writer_init_failed path=%s",
                temp_output_path,
            )
            return False

        self._runtime = _IncidentRuntime(
            writer=writer,
            temp_output_path=temp_output_path,
            started_at_ts=timestamp,
            started_at_iso=started_at_iso,
            fps=self._effective_fps(fps),
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        self._update_runtime_membership(active_alert_tracks, timestamp)
        self._logger.info(
            "alert_recording_started path=%s zones=%s tracks=%s",
            temp_output_path,
            sorted(self._runtime.zone_names_seen),
            sorted(self._runtime.display_numbers_seen or self._runtime.track_ids_seen),
        )
        return True

    def _update_runtime_membership(
        self,
        active_alert_tracks: list[ActiveAlertTrack],
        timestamp: float,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return

        active_track_ids = {track.track_id for track in active_alert_tracks}
        runtime.active_zone_names = tuple(sorted({track.zone_name for track in active_alert_tracks}))
        if active_alert_tracks:
            runtime.postbuffer_until_ts = None

        for alert_track in active_alert_tracks:
            runtime.zone_names_seen.add(alert_track.zone_name)
            runtime.track_ids_seen.add(alert_track.track_id)
            if alert_track.display_number is not None:
                runtime.display_numbers_seen.add(alert_track.display_number)
            runtime.track_display_numbers[alert_track.track_id] = alert_track.display_number
            runtime.track_zone_names.setdefault(alert_track.track_id, set()).add(alert_track.zone_name)

            if alert_track.track_id not in runtime.track_entered_at:
                entered_at_ts = alert_track.entered_at_ts or timestamp
                runtime.track_entered_at[alert_track.track_id] = entered_at_ts
                runtime.track_first_entered_at.setdefault(alert_track.track_id, entered_at_ts)

        for track_id in list(runtime.track_entered_at):
            if track_id in active_track_ids:
                continue
            entered_at_ts = runtime.track_entered_at.pop(track_id)
            runtime.track_dwell_seconds[track_id] = (
                runtime.track_dwell_seconds.get(track_id, 0.0)
                + max(0.0, timestamp - entered_at_ts)
            )
            runtime.track_last_exited_at[track_id] = timestamp

    def _build_recording_state(
        self,
        timestamp: float,
        active_alert_tracks: list[ActiveAlertTrack],
    ) -> AlertRecordingState | None:
        runtime = self._runtime
        if runtime is None:
            return None

        active_track_ids = tuple(sorted(track.track_id for track in active_alert_tracks))
        active_display_numbers = tuple(
            sorted(
                {
                    track.display_number
                    for track in active_alert_tracks
                    if track.display_number is not None
                },
            ),
        )
        zone_names = tuple(sorted({track.zone_name for track in active_alert_tracks}))
        if not zone_names:
            zone_names = runtime.active_zone_names or tuple(sorted(runtime.zone_names_seen))

        current_wallclock_iso = ""
        if self._config.include_wallclock_timestamp:
            current_wallclock_iso = self._format_wallclock(timestamp)

        return AlertRecordingState(
            output_path=str(runtime.temp_output_path),
            started_at_ts=runtime.started_at_ts,
            started_at_iso=runtime.started_at_iso,
            current_wallclock_iso=current_wallclock_iso,
            zone_names=zone_names if self._config.include_zone_name else (),
            active_track_ids=active_track_ids if self._config.include_track_ids else (),
            active_display_numbers=active_display_numbers if self._config.include_track_ids else (),
            incident_track_ids=tuple(sorted(runtime.track_ids_seen)),
            incident_display_numbers=tuple(sorted(runtime.display_numbers_seen)),
            elapsed_seconds=(timestamp - runtime.started_at_ts) if self._config.include_elapsed_seconds else 0.0,
            frame_count=runtime.frame_count,
            is_recording=True,
            postbuffer_active=not bool(active_alert_tracks),
        )

    def _write_frame(self, frame: np.ndarray, timestamp: float) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.writer.write(frame)
        runtime.frame_count += 1
        runtime.last_written_ts = timestamp

    def _append_prebuffer(self, frame: np.ndarray, timestamp: float) -> None:
        if self._config.prebuffer_seconds <= 0:
            self._prebuffer.clear()
            return
        self._prebuffer.append(_BufferedFrame(timestamp=timestamp, frame=frame.copy()))
        self._prune_prebuffer(timestamp)

    def _flush_prebuffer(self) -> None:
        runtime = self._runtime
        if runtime is None:
            self._prebuffer.clear()
            return

        while self._prebuffer:
            buffered_frame = self._prebuffer.popleft()
            runtime.writer.write(buffered_frame.frame)
            runtime.frame_count += 1
            runtime.last_written_ts = buffered_frame.timestamp

    def _prune_prebuffer(self, timestamp: float) -> None:
        cutoff_ts = timestamp - self._config.prebuffer_seconds
        while self._prebuffer and self._prebuffer[0].timestamp < cutoff_ts:
            self._prebuffer.popleft()

    def _finalize_incident(self, timestamp: float) -> None:
        runtime = self._runtime
        if runtime is None:
            return

        for track_id, entered_at_ts in list(runtime.track_entered_at.items()):
            runtime.track_dwell_seconds[track_id] = (
                runtime.track_dwell_seconds.get(track_id, 0.0)
                + max(0.0, timestamp - entered_at_ts)
            )
            runtime.track_last_exited_at[track_id] = timestamp
        runtime.track_entered_at.clear()

        runtime.writer.release()
        final_output_path = self._resolve_final_output_path(runtime)
        runtime.temp_output_path.rename(final_output_path)
        metadata_path = final_output_path.with_suffix(".json")

        incident = AlertIncident(
            video_path=str(final_output_path),
            metadata_path=str(metadata_path),
            started_at_ts=runtime.started_at_ts,
            ended_at_ts=timestamp,
            started_at_iso=runtime.started_at_iso,
            ended_at_iso=self._format_wallclock(timestamp),
            zone_names=tuple(sorted(runtime.zone_names_seen)),
            track_ids=tuple(sorted(runtime.track_ids_seen)),
            display_numbers=tuple(sorted(runtime.display_numbers_seen)),
            duration_seconds=max(0.0, timestamp - runtime.started_at_ts),
            total_frames=runtime.frame_count,
            dwell_seconds_by_track_id=dict(runtime.track_dwell_seconds),
            first_entered_at_by_track_id=dict(runtime.track_first_entered_at),
            last_exited_at_by_track_id=dict(runtime.track_last_exited_at),
            zone_names_by_track_id={
                track_id: tuple(sorted(zone_names))
                for track_id, zone_names in runtime.track_zone_names.items()
            },
        )
        self._write_metadata(incident, runtime)
        self._last_incident = incident
        self._runtime = None
        self._current_state = None
        self._logger.info(
            "alert_recording_finished video=%s duration=%.2f zones=%s tracks=%s",
            incident.video_path,
            incident.duration_seconds,
            list(incident.zone_names),
            list(incident.display_numbers or incident.track_ids),
        )

    def _write_metadata(self, incident: AlertIncident, runtime: _IncidentRuntime) -> None:
        metadata_payload = {
            "video_path": incident.video_path,
            "started_at": incident.started_at_iso,
            "ended_at": incident.ended_at_iso,
            "duration_seconds": round(incident.duration_seconds, 3),
            "total_frames": incident.total_frames,
            "fps": runtime.fps,
            "frame_size": {
                "width": runtime.frame_width,
                "height": runtime.frame_height,
            },
            "zones": list(incident.zone_names),
            "track_ids": list(incident.track_ids),
            "display_numbers": list(incident.display_numbers),
            "tracks": {
                str(track_id): {
                    "display_number": runtime.track_display_numbers.get(track_id),
                    "zones": list(incident.zone_names_by_track_id.get(track_id, ())),
                    "dwell_seconds": round(incident.dwell_seconds_by_track_id.get(track_id, 0.0), 3),
                    "first_entered_at": (
                        self._format_wallclock(incident.first_entered_at_by_track_id[track_id])
                        if track_id in incident.first_entered_at_by_track_id
                        else None
                    ),
                    "last_exited_at": (
                        self._format_wallclock(incident.last_exited_at_by_track_id[track_id])
                        if track_id in incident.last_exited_at_by_track_id
                        else None
                    ),
                }
                for track_id in incident.track_ids
            },
        }
        metadata_path = Path(incident.metadata_path)
        metadata_path.write_text(
            json.dumps(metadata_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _resolve_final_output_path(self, runtime: _IncidentRuntime) -> Path:
        output_dir = Path(self._config.output_dir).expanduser()
        timestamp_slug = datetime.fromtimestamp(runtime.started_at_ts).strftime("%Y-%m-%d_%H-%M-%S")

        zone_names = sorted(runtime.zone_names_seen)
        if len(zone_names) == 1:
            zone_segment = f"zone-{self._slugify(zone_names[0])}"
        else:
            zone_segment = "zones-" + "-".join(self._slugify(zone_name) for zone_name in zone_names)

        track_numbers = sorted(runtime.display_numbers_seen)
        if not track_numbers:
            track_numbers = sorted(runtime.track_ids_seen)
        track_segment = "-".join(str(track_number) for track_number in track_numbers) or "unknown"

        final_output_path = output_dir / f"{timestamp_slug}_{zone_segment}_tracks-{track_segment}.mp4"
        return self._resolve_unique_path(final_output_path)

    def _resolve_unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        suffix = 2
        while True:
            candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            if not candidate.exists():
                return candidate
            suffix += 1

    def _create_video_writer(
        self,
        output_path: Path,
        frame_width: int,
        frame_height: int,
        fps: float,
    ) -> cv2.VideoWriter | None:
        effective_fps = self._effective_fps(fps)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*self._config.codec),
            effective_fps,
            (frame_width, frame_height),
        )
        if writer.isOpened():
            return writer
        writer.release()
        return None

    def _effective_fps(self, fps: float) -> float:
        if fps > 0 and math.isfinite(fps):
            return fps
        return self._config.fps_fallback

    @staticmethod
    def _format_wallclock(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _slugify(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        return normalized or "zone"
