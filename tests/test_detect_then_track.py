import logging

import numpy as np

from app.models import Detection, TrackingConfig, TrackingPipelineState
from app.tracker import DetectThenTrackManager, FrameTrackerWrapper


class _FakeFrameTracker(FrameTrackerWrapper):
    def __init__(self, updates: list[tuple[bool, tuple[float, float, float, float]]]) -> None:
        self._updates = list(updates)
        self._last_bbox = (10.0, 10.0, 30.0, 30.0)

    def initialize(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
        self._last_bbox = bbox
        return True

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        if self._updates:
            ok, bbox = self._updates.pop(0)
            if ok:
                self._last_bbox = bbox
            return ok, bbox
        return True, self._last_bbox


class _FakeFrameTrackerFactory:
    def __init__(self, sequences: list[list[tuple[bool, tuple[float, float, float, float]]]]) -> None:
        self._sequences = list(sequences)

    def __call__(self) -> _FakeFrameTracker:
        updates = self._sequences.pop(0) if self._sequences else []
        return _FakeFrameTracker(updates)


def _make_manager(
    factory: _FakeFrameTrackerFactory,
    **overrides: object,
) -> DetectThenTrackManager:
    defaults: dict[str, object] = {
        "confirm_frames": 1,
        "reconfirm_frames": 1,
        "tracker_only_mode_after_confirm": True,
        "detector_interval_while_tracking": 10,
        "reacquire_after_failed_tracker_frames": 2,
        "max_tracker_only_frames_without_detection": 20,
        "hold_without_detection_frames": 10,
        "lost_transition_frames": 5,
        "max_missing_frames": 20,
        "hard_remove_frames": 30,
    }
    defaults.update(overrides)
    config = TrackingConfig(**defaults)
    return DetectThenTrackManager(
        config=config,
        logger=logging.getLogger("test_detect_then_track"),
        frame_tracker_factory=factory,
    )


def _make_detection(
    x1: float = 10.0,
    y1: float = 10.0,
    x2: float = 30.0,
    y2: float = 30.0,
    confidence: float = 0.90,
) -> Detection:
    return Detection(15, "cat", confidence, x1, y1, x2, y2)


def test_manager_switches_to_track_only_after_confirmation() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager = _make_manager(factory=_FakeFrameTrackerFactory([[]]))

    first = manager.update(
        frame=frame,
        frame_index=1,
        timestamp=1.0,
        detections=[_make_detection()],
    )
    second = manager.update(
        frame=frame,
        frame_index=2,
        timestamp=2.0,
        detections=None,
    )

    assert first.pipeline_state == TrackingPipelineState.TRACK_ONLY
    assert manager.should_run_detector(frame_index=2, detection_cycle_due=False) is False
    assert second.pipeline_state == TrackingPipelineState.TRACK_ONLY
    assert second.tracker_updates_count == 1
    assert second.yolo_ran_this_frame is False


def test_manager_requests_reacquire_after_tracker_failures() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager = _make_manager(
        factory=_FakeFrameTrackerFactory(
            [[(False, (10.0, 10.0, 30.0, 30.0)), (False, (10.0, 10.0, 30.0, 30.0))]],
        ),
        reacquire_after_failed_tracker_frames=2,
        detector_interval_while_tracking=50,
    )

    manager.update(
        frame=frame,
        frame_index=1,
        timestamp=1.0,
        detections=[_make_detection()],
    )
    manager.update(
        frame=frame,
        frame_index=2,
        timestamp=2.0,
        detections=None,
    )
    third = manager.update(
        frame=frame,
        frame_index=3,
        timestamp=3.0,
        detections=None,
    )

    assert manager.should_run_detector(frame_index=4, detection_cycle_due=False) is True
    assert third.pipeline_state == TrackingPipelineState.REACQUIRE
    assert third.tracker_failures_count == 1


def test_manager_reacquires_same_track_id_after_tracker_loss() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager = _make_manager(
        factory=_FakeFrameTrackerFactory([[(False, (10.0, 10.0, 30.0, 30.0))], []]),
        reacquire_after_failed_tracker_frames=1,
        detector_interval_while_tracking=50,
    )

    first = manager.update(
        frame=frame,
        frame_index=1,
        timestamp=1.0,
        detections=[_make_detection()],
    )
    track_id = first.visible_tracks[0].track_id

    manager.update(
        frame=frame,
        frame_index=2,
        timestamp=2.0,
        detections=None,
    )
    third = manager.update(
        frame=frame,
        frame_index=3,
        timestamp=3.0,
        detections=[_make_detection(x1=12.0, y1=10.0, x2=32.0, y2=30.0, confidence=0.20)],
    )

    assert third.visible_tracks[0].track_id == track_id
    assert third.pipeline_state == TrackingPipelineState.TRACK_ONLY
    assert third.yolo_ran_this_frame is True


def test_manager_forces_detector_after_long_tracker_only_window() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager = _make_manager(
        factory=_FakeFrameTrackerFactory([[]]),
        detector_interval_while_tracking=50,
        max_tracker_only_frames_without_detection=3,
    )

    manager.update(
        frame=frame,
        frame_index=1,
        timestamp=1.0,
        detections=[_make_detection()],
    )
    manager.update(
        frame=frame,
        frame_index=2,
        timestamp=2.0,
        detections=None,
    )
    manager.update(
        frame=frame,
        frame_index=3,
        timestamp=3.0,
        detections=None,
    )

    assert manager.should_run_detector(frame_index=4, detection_cycle_due=False) is True


def test_manager_does_not_create_parallel_duplicate_for_same_cat_after_tracker_drift() -> None:
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    manager = _make_manager(
        factory=_FakeFrameTrackerFactory([[(True, (40.0, 10.0, 60.0, 30.0))]]),
        detector_interval_while_tracking=50,
        reconfirm_frames=1,
        smoothing_alpha=1.0,
    )

    first = manager.update(
        frame=frame,
        frame_index=1,
        timestamp=1.0,
        detections=[_make_detection()],
    )
    track_id = first.visible_tracks[0].track_id

    manager.update(
        frame=frame,
        frame_index=2,
        timestamp=2.0,
        detections=None,
    )
    third = manager.update(
        frame=frame,
        frame_index=3,
        timestamp=3.0,
        detections=[_make_detection(x1=14.0, y1=10.0, x2=34.0, y2=30.0, confidence=0.70)],
    )

    assert third.confirmed_count == 1
    assert third.tentative_tracks == []
    assert len({track.track_id for track in third.visible_tracks}) == 1
    assert third.visible_tracks[0].track_id == track_id
