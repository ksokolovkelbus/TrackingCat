import logging
from unittest.mock import patch

from app.main import process_detections
from app.models import AppConfig, Detection, TrackState, TrackingConfig
from app.target_selector import TargetSelector
from app.tracker import MultiCatTracker, OpenCvFrameTracker


def _make_tracker(**overrides: object) -> MultiCatTracker:
    defaults: dict[str, object] = {
        "confirm_frames": 1,
        "hold_without_detection_frames": 2,
        "lost_transition_frames": 2,
        "max_missing_frames": 6,
        "hard_remove_frames": 12,
    }
    defaults.update(overrides)
    config = TrackingConfig(**defaults)
    return MultiCatTracker(config=config, logger=logging.getLogger("test_tracker"))


class _TrackerSpy:
    def __init__(self) -> None:
        self.called = False

    def update(self, *args: object, **kwargs: object) -> None:
        self.called = True
        raise AssertionError("Tracker must not be called in detection-only mode.")


def test_tracking_disabled_false_uses_detection_only_mode_without_tracks() -> None:
    app_config = AppConfig()
    app_config.tracking.tracking_enabled = False
    selector = TargetSelector(strategy=app_config.target_selection_strategy)
    tracker_spy = _TrackerSpy()

    summary = process_detections(
        detections=[
            Detection(15, "cat", 0.90, 10, 10, 30, 30),
            Detection(15, "cat", 0.85, 40, 10, 60, 30),
        ],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
        config=app_config,
        selector=selector,
        tracker=tracker_spy,  # type: ignore[arg-type]
    )

    assert tracker_spy.called is False
    assert summary.tracking_enabled is False
    assert summary.active_tracks_count == 0
    assert summary.visible_tracks == []
    assert summary.detection_overlay_count == 2


def test_confirmed_track_survives_one_bad_frame() -> None:
    tracker = _make_tracker()

    tracker.update(
        detections=[Detection(15, "cat", 0.90, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    summary = tracker.update(
        detections=[],
        frame_shape=(100, 100, 3),
        frame_index=2,
        timestamp=2.0,
    )

    assert summary.visible_count == 1
    assert summary.visible_tracks[0].state == TrackState.HELD


def test_confirmed_track_is_not_removed_after_short_detection_gap() -> None:
    tracker = _make_tracker(max_missing_frames=6, hard_remove_frames=10)

    first = tracker.update(
        detections=[Detection(15, "cat", 0.90, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    for frame_index in range(2, 6):
        summary = tracker.update(
            detections=[],
            frame_shape=(100, 100, 3),
            frame_index=frame_index,
            timestamp=float(frame_index),
        )

    assert summary.removed_track_ids == []
    assert summary.visible_tracks[0].track_id == first.visible_tracks[0].track_id
    assert summary.visible_tracks[0].state == TrackState.HELD


def test_low_confidence_nearby_detection_keeps_same_track() -> None:
    tracker = _make_tracker(
        acquire_confidence_threshold=0.35,
        keep_confidence_threshold=0.10,
    )

    first = tracker.update(
        detections=[Detection(15, "cat", 0.88, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    second = tracker.update(
        detections=[Detection(15, "cat", 0.11, 12, 11, 32, 31)],
        frame_shape=(100, 100, 3),
        frame_index=2,
        timestamp=2.0,
    )

    assert second.visible_count == 1
    assert second.visible_tracks[0].track_id == first.visible_tracks[0].track_id
    assert second.visible_tracks[0].state == TrackState.CONFIRMED


def test_held_track_returns_to_confirmed_without_changing_track_id() -> None:
    tracker = _make_tracker(reconfirm_frames=2)

    first = tracker.update(
        detections=[Detection(15, "cat", 0.90, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    tracker.update(
        detections=[],
        frame_shape=(100, 100, 3),
        frame_index=2,
        timestamp=2.0,
    )
    third = tracker.update(
        detections=[Detection(15, "cat", 0.12, 11, 10, 31, 30)],
        frame_shape=(100, 100, 3),
        frame_index=3,
        timestamp=3.0,
    )
    fourth = tracker.update(
        detections=[Detection(15, "cat", 0.13, 12, 10, 32, 30)],
        frame_shape=(100, 100, 3),
        frame_index=4,
        timestamp=4.0,
    )

    assert third.visible_tracks[0].state == TrackState.HELD
    assert fourth.visible_tracks[0].state == TrackState.CONFIRMED
    assert fourth.visible_tracks[0].track_id == first.visible_tracks[0].track_id


def test_no_new_track_is_created_while_old_confirmed_track_can_be_held() -> None:
    tracker = _make_tracker(reconfirm_frames=2)

    first = tracker.update(
        detections=[Detection(15, "cat", 0.92, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    tracker.update(
        detections=[],
        frame_shape=(100, 100, 3),
        frame_index=2,
        timestamp=2.0,
    )
    summary = tracker.update(
        detections=[Detection(15, "cat", 0.60, 12, 10, 32, 30)],
        frame_shape=(100, 100, 3),
        frame_index=3,
        timestamp=3.0,
    )

    assert len(summary.visible_tracks) == 1
    assert len(summary.tentative_tracks) == 0
    assert summary.visible_tracks[0].track_id == first.visible_tracks[0].track_id


def test_tracker_does_not_oscillate_between_lost_and_reacquired_every_frame() -> None:
    tracker = _make_tracker(
        reconfirm_frames=2,
        max_missing_frames=8,
        hard_remove_frames=12,
    )

    states: list[TrackState] = []
    tracker.update(
        detections=[Detection(15, "cat", 0.90, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    for frame_index, detections in [
        (2, []),
        (3, [Detection(15, "cat", 0.12, 11, 10, 31, 30)]),
        (4, []),
        (5, [Detection(15, "cat", 0.12, 12, 10, 32, 30)]),
        (6, []),
        (7, [Detection(15, "cat", 0.12, 13, 10, 33, 30)]),
    ]:
        summary = tracker.update(
            detections=detections,
            frame_shape=(100, 100, 3),
            frame_index=frame_index,
            timestamp=float(frame_index),
        )
        if summary.visible_tracks:
            states.append(summary.visible_tracks[0].state)
        elif summary.lost_tracks:
            states.append(summary.lost_tracks[0].state)

    assert TrackState.LOST not in states
    assert all(state in {TrackState.HELD, TrackState.CONFIRMED} for state in states)


def test_track_is_removed_only_after_long_real_loss() -> None:
    tracker = _make_tracker(max_missing_frames=4, hard_remove_frames=7)

    tracker.update(
        detections=[Detection(15, "cat", 0.90, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )

    removed_frames: list[int] = []
    for frame_index in range(2, 9):
        summary = tracker.update(
            detections=[],
            frame_shape=(100, 100, 3),
            frame_index=frame_index,
            timestamp=float(frame_index),
        )
        if summary.removed_track_ids:
            removed_frames.append(frame_index)

    assert removed_frames == [8]


def test_multi_target_tracking_keeps_three_real_cats_as_three_tracks() -> None:
    tracker = _make_tracker(confirm_frames=1)

    first = tracker.update(
        detections=[
            Detection(15, "cat", 0.95, 10, 10, 30, 30),
            Detection(15, "cat", 0.93, 80, 12, 100, 32),
            Detection(15, "cat", 0.91, 150, 14, 170, 34),
        ],
        frame_shape=(200, 200, 3),
        frame_index=1,
        timestamp=1.0,
    )
    second = tracker.update(
        detections=[
            Detection(15, "cat", 0.94, 12, 10, 32, 30),
            Detection(15, "cat", 0.92, 82, 12, 102, 32),
            Detection(15, "cat", 0.90, 152, 14, 172, 34),
        ],
        frame_shape=(200, 200, 3),
        frame_index=2,
        timestamp=2.0,
    )

    assert first.visible_count == 3
    assert second.visible_count == 3
    assert [track.track_id for track in second.visible_tracks] == [1, 2, 3]


def test_lost_track_reacquires_same_id_within_reacquire_window() -> None:
    tracker = _make_tracker(
        confirm_frames=1,
        reconfirm_frames=1,
        hold_without_detection_frames=1,
        lost_transition_frames=1,
        max_missing_frames=2,
        hard_remove_frames=8,
        reacquire_max_frames=3,
    )

    first = tracker.update(
        detections=[Detection(15, "cat", 0.92, 10, 10, 30, 30)],
        frame_shape=(100, 100, 3),
        frame_index=1,
        timestamp=1.0,
    )
    track_id = first.visible_tracks[0].track_id

    tracker.update(
        detections=[],
        frame_shape=(100, 100, 3),
        frame_index=2,
        timestamp=2.0,
    )
    lost = tracker.update(
        detections=[],
        frame_shape=(100, 100, 3),
        frame_index=3,
        timestamp=3.0,
    )
    reacquired = tracker.update(
        detections=[Detection(15, "cat", 0.30, 12, 10, 32, 30)],
        frame_shape=(100, 100, 3),
        frame_index=4,
        timestamp=4.0,
    )

    assert lost.lost_count == 1
    assert reacquired.visible_tracks[0].track_id == track_id
    assert reacquired.visible_tracks[0].state == TrackState.CONFIRMED


def test_real_new_cat_gets_new_track_id_without_replacing_existing_cat() -> None:
    tracker = _make_tracker(confirm_frames=1)

    first = tracker.update(
        detections=[Detection(15, "cat", 0.93, 10, 10, 30, 30)],
        frame_shape=(200, 200, 3),
        frame_index=1,
        timestamp=1.0,
    )
    summary = tracker.update(
        detections=[
            Detection(15, "cat", 0.94, 12, 10, 32, 30),
            Detection(15, "cat", 0.92, 120, 80, 150, 110),
        ],
        frame_shape=(200, 200, 3),
        frame_index=2,
        timestamp=2.0,
    )

    assert summary.visible_count == 2
    assert {track.track_id for track in summary.visible_tracks} == {
        first.visible_tracks[0].track_id,
        first.visible_tracks[0].track_id + 1,
    }


def test_duplicate_detection_is_reserved_for_existing_confirmed_track() -> None:
    tracker = _make_tracker(confirm_frames=1, soft_center_distance_gate=180.0)

    tracker.update(
        detections=[Detection(15, "cat", 0.95, 10, 10, 30, 30)],
        frame_shape=(200, 200, 3),
        frame_index=1,
        timestamp=1.0,
    )
    summary = tracker.update(
        detections=[
            Detection(15, "cat", 0.96, 12, 10, 32, 30),
            Detection(15, "cat", 0.90, 13, 11, 33, 31),
        ],
        frame_shape=(200, 200, 3),
        frame_index=2,
        timestamp=2.0,
    )

    assert summary.confirmed_count == 1
    assert summary.tentative_tracks == []


def test_opencv_frame_tracker_prefers_csrt_backend() -> None:
    factory_calls: list[str] = []

    def resolve(backend_name: str) -> object:
        factory_calls.append(backend_name)
        if backend_name == "CSRT":
            return lambda: object()
        return None

    with patch.object(OpenCvFrameTracker, "_resolve_backend_factory", side_effect=resolve):
        tracker = OpenCvFrameTracker()

    assert tracker.backend_name == "CSRT"
    assert factory_calls == ["CSRT"]


def test_opencv_frame_tracker_falls_back_to_kcf_backend() -> None:
    factory_calls: list[str] = []

    def resolve(backend_name: str) -> object:
        factory_calls.append(backend_name)
        if backend_name == "KCF":
            return lambda: object()
        return None

    with patch.object(OpenCvFrameTracker, "_resolve_backend_factory", side_effect=resolve):
        tracker = OpenCvFrameTracker()

    assert tracker.backend_name == "KCF"
    assert factory_calls == ["CSRT", "KCF"]


def test_opencv_frame_tracker_falls_back_to_mil_backend() -> None:
    factory_calls: list[str] = []

    def resolve(backend_name: str) -> object:
        factory_calls.append(backend_name)
        if backend_name == "MIL":
            return lambda: object()
        return None

    with patch.object(OpenCvFrameTracker, "_resolve_backend_factory", side_effect=resolve):
        tracker = OpenCvFrameTracker()

    assert tracker.backend_name == "MIL"
    assert factory_calls == ["CSRT", "KCF", "MIL"]
