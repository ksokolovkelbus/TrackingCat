from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.alert_recorder import AlertRecorder
from app.models import ActiveAlertTrack, AlertRecordingConfig, ZoneType


class _FakeVideoWriter:
    instances: list["_FakeVideoWriter"] = []

    def __init__(self, path: str, _fourcc: int, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.fps = fps
        self.size = size
        self.frames: list[np.ndarray] = []
        self.released = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.__class__.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def _make_frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _make_alert_track(
    track_id: int,
    display_number: int,
    zone_name: str = "restricted_1",
    zone_type: ZoneType = ZoneType.RESTRICTED,
    entered_at_ts: float = 0.0,
) -> ActiveAlertTrack:
    return ActiveAlertTrack(
        track_id=track_id,
        display_number=display_number,
        zone_name=zone_name,
        zone_type=zone_type,
        entered_at_ts=entered_at_ts,
        dwell_seconds=0.0,
    )


def _make_recorder(output_dir: Path, **overrides: object) -> AlertRecorder:
    config = AlertRecordingConfig(
        output_dir=str(output_dir),
        fps_fallback=2.0,
        draw_recording_overlay=False,
        **overrides,
    )
    return AlertRecorder(config=config, logger=logging.getLogger("test_alert_recorder"))


def test_incident_recording_starts_on_first_alert_and_includes_prebuffer(tmp_path) -> None:
    _FakeVideoWriter.instances.clear()
    recorder = _make_recorder(tmp_path / "alerts", prebuffer_seconds=1.0, postbuffer_seconds=0.5)

    with (
        patch("app.alert_recorder.cv2.VideoWriter", _FakeVideoWriter),
        patch("app.alert_recorder.cv2.VideoWriter_fourcc", return_value=0),
    ):
        recorder.process_frame(_make_frame(), frame_index=1, timestamp=0.0, fps=2.0, active_alert_tracks=[])
        recorder.process_frame(_make_frame(), frame_index=2, timestamp=0.4, fps=2.0, active_alert_tracks=[])
        state = recorder.process_frame(
            _make_frame(),
            frame_index=3,
            timestamp=0.8,
            fps=2.0,
            active_alert_tracks=[_make_alert_track(3, 3, entered_at_ts=0.8)],
        )

    writer = _FakeVideoWriter.instances[0]
    assert state is not None
    assert len(writer.frames) == 3
    assert writer.path.parent.exists()


def test_incident_recording_keeps_writing_while_alert_active_and_stops_after_postbuffer(tmp_path) -> None:
    _FakeVideoWriter.instances.clear()
    recorder = _make_recorder(tmp_path / "alerts", prebuffer_seconds=0.0, postbuffer_seconds=0.5)

    with (
        patch("app.alert_recorder.cv2.VideoWriter", _FakeVideoWriter),
        patch("app.alert_recorder.cv2.VideoWriter_fourcc", return_value=0),
    ):
        recorder.process_frame(
            _make_frame(),
            frame_index=1,
            timestamp=1.0,
            fps=2.0,
            active_alert_tracks=[_make_alert_track(3, 3, entered_at_ts=1.0)],
        )
        recorder.process_frame(
            _make_frame(),
            frame_index=2,
            timestamp=1.3,
            fps=2.0,
            active_alert_tracks=[_make_alert_track(3, 3, entered_at_ts=1.0)],
        )
        state = recorder.process_frame(
            _make_frame(),
            frame_index=3,
            timestamp=1.6,
            fps=2.0,
            active_alert_tracks=[],
        )
        assert state is not None
        recorder.process_frame(
            _make_frame(),
            frame_index=4,
            timestamp=1.9,
            fps=2.0,
            active_alert_tracks=[],
        )
        state = recorder.process_frame(
            _make_frame(),
            frame_index=5,
            timestamp=2.2,
            fps=2.0,
            active_alert_tracks=[],
        )

    writer = _FakeVideoWriter.instances[0]
    assert len(writer.frames) == 5
    assert writer.released is True
    assert state is None
    assert recorder.last_incident is not None


def test_output_dir_is_auto_created_and_filename_contains_timestamp_zone_and_tracks(tmp_path) -> None:
    _FakeVideoWriter.instances.clear()
    output_dir = tmp_path / "alert_video"
    recorder = _make_recorder(output_dir, prebuffer_seconds=0.0, postbuffer_seconds=0.0)
    started_at_ts = datetime(2026, 4, 20, 20, 10, 31).timestamp()

    with (
        patch("app.alert_recorder.cv2.VideoWriter", _FakeVideoWriter),
        patch("app.alert_recorder.cv2.VideoWriter_fourcc", return_value=0),
    ):
        recorder.process_frame(
            _make_frame(),
            frame_index=1,
            timestamp=started_at_ts,
            fps=2.0,
            active_alert_tracks=[
                _make_alert_track(1, 1, entered_at_ts=started_at_ts),
                _make_alert_track(3, 3, entered_at_ts=started_at_ts),
            ],
        )
        recorder.process_frame(
            _make_frame(),
            frame_index=2,
            timestamp=started_at_ts + 0.1,
            fps=2.0,
            active_alert_tracks=[],
        )

    assert output_dir.exists()
    assert recorder.last_incident is not None
    final_path = Path(recorder.last_incident.video_path)
    assert "2026-04-20_20-10-31" in final_path.name
    assert "zone-restricted_1" in final_path.name
    assert "tracks-1-3" in final_path.name
    assert Path(recorder.last_incident.metadata_path).exists()


def test_multi_cat_incident_uses_single_recording_session_while_any_cat_remains_in_zone(tmp_path) -> None:
    _FakeVideoWriter.instances.clear()
    recorder = _make_recorder(tmp_path / "alerts", prebuffer_seconds=0.0, postbuffer_seconds=0.2)

    with (
        patch("app.alert_recorder.cv2.VideoWriter", _FakeVideoWriter),
        patch("app.alert_recorder.cv2.VideoWriter_fourcc", return_value=0),
    ):
        recorder.process_frame(
            _make_frame(),
            frame_index=1,
            timestamp=1.0,
            fps=2.0,
            active_alert_tracks=[_make_alert_track(1, 1, entered_at_ts=1.0)],
        )
        recorder.process_frame(
            _make_frame(),
            frame_index=2,
            timestamp=1.2,
            fps=2.0,
            active_alert_tracks=[
                _make_alert_track(1, 1, entered_at_ts=1.0),
                _make_alert_track(3, 3, entered_at_ts=1.2),
            ],
        )
        state = recorder.process_frame(
            _make_frame(),
            frame_index=3,
            timestamp=1.4,
            fps=2.0,
            active_alert_tracks=[_make_alert_track(3, 3, entered_at_ts=1.2)],
        )
        recorder.process_frame(
            _make_frame(),
            frame_index=4,
            timestamp=1.7,
            fps=2.0,
            active_alert_tracks=[],
        )
        recorder.process_frame(
            _make_frame(),
            frame_index=5,
            timestamp=2.0,
            fps=2.0,
            active_alert_tracks=[],
        )

    assert state is not None
    assert len(_FakeVideoWriter.instances) == 1
    assert recorder.last_incident is not None
    assert recorder.last_incident.track_ids == (1, 3)
