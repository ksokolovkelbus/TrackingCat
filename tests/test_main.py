import logging

from app.main import _adjust_detector_threshold_for_runtime_mode, _build_config_summary, _dump_effective_config, _should_process_frame
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


def test_dump_effective_config_includes_expected_values() -> None:
    config = AppConfig()
    config.output.show_window = False
    dumped = _dump_effective_config(config)

    assert "output:" in dumped
    assert "show_window: false" in dumped


def test_build_config_summary_contains_key_runtime_settings() -> None:
    config = AppConfig()
    config.source.source_type = "http_snapshot"
    config.source.stream_url = "http://cam.local/snapshot.jpg"
    config.output.show_window = False

    summary = _build_config_summary(config)

    assert "source_type=http_snapshot" in summary
    assert "source=http://cam.local/snapshot.jpg" in summary
    assert "tracking_enabled=True" in summary
    assert "show_window=False" in summary
