import logging

from app.main import _adjust_detector_threshold_for_runtime_mode, _should_process_frame
from app.models import AppConfig, FrameTrackingSummary


def test_detection_only_mode_lowers_detector_threshold_when_not_overridden() -> None:
    config = AppConfig()
    config.tracking.tracking_enabled = False
    config.detector.confidence_threshold = 0.35
    config.tracking.keep_confidence_threshold = 0.10

    _adjust_detector_threshold_for_runtime_mode(
        config=config,
        logger=logging.getLogger("test_main"),
        detector_threshold_overridden=False,
    )

    assert config.detector.confidence_threshold == 0.10


def test_explicit_detector_threshold_is_not_overridden() -> None:
    config = AppConfig()
    config.tracking.tracking_enabled = False
    config.detector.confidence_threshold = 0.35
    config.tracking.keep_confidence_threshold = 0.10

    _adjust_detector_threshold_for_runtime_mode(
        config=config,
        logger=logging.getLogger("test_main"),
        detector_threshold_overridden=True,
    )

    assert config.detector.confidence_threshold == 0.35


def test_should_process_frame_respects_process_every_n_frames() -> None:
    summary = FrameTrackingSummary(
        frame_index=1,
        frame_width=1280,
        frame_height=720,
        tracking_enabled=False,
        detections_count=0,
    )

    assert _should_process_frame(frame_index=1, process_every_n_frames=3, last_summary=None) is True
    assert _should_process_frame(frame_index=2, process_every_n_frames=3, last_summary=summary) is False
    assert _should_process_frame(frame_index=3, process_every_n_frames=3, last_summary=summary) is False
    assert _should_process_frame(frame_index=4, process_every_n_frames=3, last_summary=summary) is True
