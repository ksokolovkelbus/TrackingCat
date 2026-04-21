from __future__ import annotations

import argparse
import logging
import signal
import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from app.config import ConfigError, load_config, parse_bool
from app.alert_recorder import AlertRecorder
from app.audio_alert import AudioAlertPlayer
from app.detector import DetectorError, YOLODetector
from app.logger_setup import TrackCoordinateLogger, setup_logging
from app.models import AppConfig, Detection, FrameTrackingSummary, TrackingPipelineState
from app.overlay import OverlayRenderer
from app.surface_monitor import SurfaceMonitor
from app.target_selector import TargetSelector
from app.tracker import DetectThenTrackManager, MultiCatTracker
from app.utils import FPSMeter, build_detection_overlays
from app.video_source import VideoSource, VideoSourceError
from app.zone_editor import ZoneEditor
from app.zones import SceneZoneClassifier


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and track cats in a video stream.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--source", choices=["webcam", "file", "rtsp", "http_snapshot", "browser_upload"], help="Input source type.")
    parser.add_argument("--input", help="Video file path or RTSP/URL stream.")
    parser.add_argument("--camera-index", type=int, help="Camera index for webcam source.")
    parser.add_argument("--camera-width", type=int, help="Requested camera capture width.")
    parser.add_argument("--camera-height", type=int, help="Requested camera capture height.")
    parser.add_argument("--snapshot-timeout", type=float, help="HTTP timeout for snapshot-based sources.")
    parser.add_argument("--browser-port", type=int, help="Port for browser camera ingest page.")
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        help="Run YOLO search/reacquire once every N frames when no confirmed tracker is active.",
    )
    parser.add_argument("--model", help="Path to YOLO model.")
    parser.add_argument("--imgsz", type=int, help="YOLO inference image size.")
    parser.add_argument("--device", help="Inference device, for example 'cpu' or 'cuda:0'.")
    parser.add_argument("--conf-thres", type=float, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, help="IoU threshold.")
    parser.add_argument(
        "--strategy",
        choices=["largest_area", "highest_confidence", "closest_to_center"],
        help="Target selection strategy for single-target compatibility mode.",
    )
    parser.add_argument("--show-window", type=parse_bool, help="Enable or disable OpenCV GUI.")
    parser.add_argument("--save-output", type=parse_bool, help="Enable or disable video saving.")
    parser.add_argument("--output-path", help="Path to processed output video.")
    parser.add_argument("--log-level", help="Logging level.")
    parser.add_argument(
        "--per-frame-debug",
        type=parse_bool,
        help="Enable per-frame debug counters in logs. Usually paired with --log-level DEBUG.",
    )
    parser.add_argument("--tracking-enabled", type=parse_bool, help="Enable tracking refinements.")
    parser.add_argument(
        "--multi-target-enabled",
        type=parse_bool,
        help="Enable multiple simultaneous cat tracks.",
    )
    parser.add_argument(
        "--tracker-only-mode-after-confirm",
        type=parse_bool,
        help="After confirmation switch from YOLO-every-frame to detect-then-track mode.",
    )
    parser.add_argument(
        "--detector-interval-while-tracking",
        type=int,
        help="Run rare YOLO validation every N frames while trackers are active.",
    )
    parser.add_argument(
        "--reacquire-after-failed-tracker-frames",
        type=int,
        help="Start YOLO reacquire after tracker fails this many consecutive frames.",
    )
    parser.add_argument(
        "--max-tracker-only-frames-without-detection",
        type=int,
        help="Force a YOLO refresh if a track lives too long without a detector correction.",
    )
    parser.add_argument(
        "--acquire-confidence-threshold",
        type=float,
        help="Confidence threshold for creating or confirming new tracks.",
    )
    parser.add_argument(
        "--keep-confidence-threshold",
        type=float,
        help="Confidence threshold for keeping confirmed tracks alive.",
    )
    parser.add_argument("--confirm-frames", type=int, help="Frames required to confirm a track.")
    parser.add_argument(
        "--reconfirm-frames",
        type=int,
        help="Frames required to recover HELD or LOST tracks.",
    )
    parser.add_argument(
        "--hold-without-detection-frames",
        type=int,
        help="Frames to keep confirmed tracks displayed without matching detections.",
    )
    parser.add_argument(
        "--lost-transition-frames",
        type=int,
        help="Additional hysteresis frames before HELD transitions to LOST.",
    )
    parser.add_argument(
        "--min-confirm-confidence",
        type=float,
        help="Backward-compatible alias for acquire threshold.",
    )
    parser.add_argument("--max-missing-frames", type=int, help="Frames to keep a lost track before removal.")
    parser.add_argument("--hard-remove-frames", type=int, help="Frames before a track is permanently removed.")
    parser.add_argument("--reacquire-max-frames", type=int, help="Frames a lost track remains matchable.")
    parser.add_argument("--iou-gate", type=float, help="Minimum IoU gate for association.")
    parser.add_argument("--center-distance-gate", type=float, help="Maximum center distance gate for association.")
    parser.add_argument("--min-area-ratio", type=float, help="Minimum detection/track area ratio gate.")
    parser.add_argument("--max-area-ratio", type=float, help="Maximum detection/track area ratio gate.")
    parser.add_argument("--soft-iou-gate", type=float, help="Soft IoU gate for confirmed or held tracks.")
    parser.add_argument(
        "--soft-center-distance-gate",
        type=float,
        help="Soft center-distance gate for confirmed or held tracks.",
    )
    parser.add_argument("--soft-min-area-ratio", type=float, help="Soft minimum area ratio for sticky matching.")
    parser.add_argument("--soft-max-area-ratio", type=float, help="Soft maximum area ratio for sticky matching.")
    parser.add_argument("--smoothing-alpha", type=float, help="EMA smoothing coefficient.")
    parser.add_argument(
        "--preserve-confirmed-tracks",
        type=parse_bool,
        help="Preserve confirmed tracks through temporary detector uncertainty.",
    )
    parser.add_argument(
        "--never-drop-confirmed-on-single-bad-frame",
        type=parse_bool,
        help="Prevent a confirmed track from dropping on one bad frame.",
    )
    parser.add_argument("--keep-lost-tracks", type=parse_bool, help="Keep lost tracks for reacquisition.")
    parser.add_argument("--max-active-tracks", type=int, help="Maximum active tracks allowed.")
    parser.add_argument("--use-motion-prediction", type=parse_bool, help="Enable simple constant velocity prediction.")
    parser.add_argument(
        "--display-sort-mode",
        choices=["top_to_bottom_left_to_right"],
        help="Display sort mode for compact visible numbering.",
    )
    parser.add_argument("--class-name", help="Class name to keep from YOLO detections.")
    parser.add_argument("--log-coordinates", type=parse_bool, help="Enable coordinate logging.")
    parser.add_argument("--coordinates-log-path", help="CSV or JSONL path for coordinate logging.")
    parser.add_argument(
        "--coordinates-log-format",
        choices=["csv", "jsonl"],
        help="Structured coordinate log format.",
    )
    parser.add_argument("--resize-enabled", type=parse_bool, help="Enable or disable frame resize.")
    parser.add_argument("--resize-width", type=int, help="Resize width.")
    parser.add_argument("--resize-height", type=int, help="Resize height.")
    parser.add_argument("--minimal-overlay", type=parse_bool, help="Draw shorter per-object labels.")
    parser.add_argument("--debug-overlay", type=parse_bool, help="Show or hide the overlay status panel.")
    parser.add_argument("--show-fps", type=parse_bool, help="Show or hide the FPS label.")
    parser.add_argument(
        "--show-normalized-coords",
        type=parse_bool,
        help="Show or hide normalized coordinates in object labels.",
    )
    parser.add_argument(
        "--show-debug-counters",
        type=parse_bool,
        help="Show or hide raw detection and tracker counters in the debug overlay panel.",
    )
    parser.add_argument(
        "--zone-editor",
        type=parse_bool,
        help="Open interactive scene zone editor instead of the tracking pipeline.",
    )
    return parser


def build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {
        "source.source_type": args.source,
        "source.camera_index": args.camera_index,
        "source.camera_width": args.camera_width,
        "source.camera_height": args.camera_height,
        "source.snapshot_timeout_seconds": args.snapshot_timeout,
        "source.browser_camera_port": args.browser_port,
        "source.process_every_n_frames": args.process_every_n_frames,
        "detector.model_path": args.model,
        "detector.imgsz": args.imgsz,
        "detector.device": args.device,
        "detector.confidence_threshold": args.conf_thres,
        "detector.iou_threshold": args.iou_thres,
        "detector.class_name": args.class_name,
        "target_selection_strategy": args.strategy,
        "overlay.minimal_overlay": args.minimal_overlay,
        "overlay.debug_overlay": args.debug_overlay,
        "overlay.show_fps": args.show_fps,
        "overlay.show_normalized_coords": args.show_normalized_coords,
        "overlay.show_debug_counters": args.show_debug_counters,
        "output.show_window": args.show_window,
        "output.save_output": args.save_output,
        "output.output_path": args.output_path,
        "logging.log_level": args.log_level.upper() if args.log_level else None,
        "logging.per_frame_debug": args.per_frame_debug,
        "logging.log_coordinates": args.log_coordinates,
        "logging.coordinates_log_path": args.coordinates_log_path,
        "logging.coordinates_log_format": args.coordinates_log_format,
        "tracking.tracking_enabled": args.tracking_enabled,
        "tracking.multi_target_enabled": args.multi_target_enabled,
        "tracking.tracker_only_mode_after_confirm": args.tracker_only_mode_after_confirm,
        "tracking.detector_interval_while_tracking": args.detector_interval_while_tracking,
        "tracking.reacquire_after_failed_tracker_frames": args.reacquire_after_failed_tracker_frames,
        "tracking.max_tracker_only_frames_without_detection": args.max_tracker_only_frames_without_detection,
        "tracking.acquire_confidence_threshold": args.acquire_confidence_threshold,
        "tracking.keep_confidence_threshold": args.keep_confidence_threshold,
        "tracking.confirm_frames": args.confirm_frames,
        "tracking.reconfirm_frames": args.reconfirm_frames,
        "tracking.hold_without_detection_frames": args.hold_without_detection_frames,
        "tracking.lost_transition_frames": args.lost_transition_frames,
        "tracking.hard_remove_frames": args.hard_remove_frames,
        "tracking.min_confirm_confidence": args.min_confirm_confidence,
        "tracking.max_missing_frames": args.max_missing_frames,
        "tracking.reacquire_max_frames": args.reacquire_max_frames,
        "tracking.iou_gate": args.iou_gate,
        "tracking.center_distance_gate": args.center_distance_gate,
        "tracking.min_area_ratio": args.min_area_ratio,
        "tracking.max_area_ratio": args.max_area_ratio,
        "tracking.soft_iou_gate": args.soft_iou_gate,
        "tracking.soft_center_distance_gate": args.soft_center_distance_gate,
        "tracking.soft_min_area_ratio": args.soft_min_area_ratio,
        "tracking.soft_max_area_ratio": args.soft_max_area_ratio,
        "tracking.smoothing_alpha": args.smoothing_alpha,
        "tracking.preserve_confirmed_tracks": args.preserve_confirmed_tracks,
        "tracking.never_drop_confirmed_on_single_bad_frame": args.never_drop_confirmed_on_single_bad_frame,
        "tracking.keep_lost_tracks": args.keep_lost_tracks,
        "tracking.max_active_tracks": args.max_active_tracks,
        "tracking.use_motion_prediction": args.use_motion_prediction,
        "tracking.display_sort_mode": args.display_sort_mode,
        "resize.enabled": args.resize_enabled,
        "resize.width": args.resize_width,
        "resize.height": args.resize_height,
        "scene_zones.zone_editor_enabled": args.zone_editor,
    }
    if args.input:
        overrides["source.source_path"] = args.input
        overrides["source.stream_url"] = args.input
    return overrides


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        config = load_config(config_path=args.config, overrides=build_cli_overrides(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2

    logger = setup_logging(config.logging.log_level)
    logger.info("Starting TrackingCat.")
    if config.scene_zones.zone_editor_enabled:
        zone_editor = ZoneEditor(
            config=config,
            config_path=args.config,
            logger=logger,
        )
        return zone_editor.run()

    _adjust_detector_threshold_for_runtime_mode(
        config=config,
        logger=logger,
        detector_threshold_overridden=args.conf_thres is not None,
    )

    try:
        detector = YOLODetector(config=config.detector, logger=logger)
        source = VideoSource(config=config.source, logger=logger)
    except (DetectorError, VideoSourceError) as exc:
        logger.error(str(exc))
        return 3

    selector = TargetSelector(strategy=config.target_selection_strategy)
    tracker_manager = (
        DetectThenTrackManager(config=config.tracking, logger=logger)
        if config.tracking.tracking_enabled
        else None
    )
    overlay = OverlayRenderer(config.overlay)
    coordinate_logger = TrackCoordinateLogger(config=config, logger=logger)
    zone_classifier = SceneZoneClassifier(config.scene_zones)
    surface_monitor = SurfaceMonitor(
        classifier=zone_classifier,
        config=config.surface_alert,
        logger=logger,
        audio_player=AudioAlertPlayer(config=config.surface_alert, logger=logger),
    )
    alert_recorder = AlertRecorder(
        config=config.alert_recording,
        logger=logger,
        overlay_renderer=overlay,
    )
    fps_meter = FPSMeter()
    video_writer: cv2.VideoWriter | None = None
    shutdown_event = Event()
    window_enabled = _initialize_window(config, logger)
    last_summary: FrameTrackingSummary | None = None

    def _handle_shutdown(signum: int, _frame: object) -> None:
        logger.info("Signal %d received. Shutting down.", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_shutdown)

    frame_index = 0

    try:
        source.open()
        while not shutdown_event.is_set():
            ok, frame = source.read()
            if not ok:
                if source.status == "ended":
                    logger.info("Input stream ended.")
                    break
                time.sleep(0.01)
                continue

            if frame is None or frame.size == 0:
                logger.warning("Received empty frame from source. Skipping.")
                continue

            frame_index += 1
            frame = _maybe_resize_frame(frame, config)
            fps = fps_meter.update()
            timestamp = time.time()
            detection_cycle_due = _should_process_frame(
                frame_index=frame_index,
                process_every_n_frames=config.source.process_every_n_frames,
                last_summary=last_summary,
            )

            if tracker_manager is None:
                summary = _process_detection_only_frame(
                    detector=detector,
                    frame=frame,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detection_cycle_due=detection_cycle_due,
                    config=config,
                    selector=selector,
                    last_summary=last_summary,
                )
            else:
                run_detector = tracker_manager.should_run_detector(
                    frame_index=frame_index,
                    detection_cycle_due=detection_cycle_due,
                )
                detections = detector.detect(frame) if run_detector else None
                selected_detections = (
                    _select_detections_for_mode(
                        detections=detections,
                        frame_shape=frame.shape,
                        config=config,
                        selector=selector,
                    )
                    if detections is not None
                    else None
                )
                summary = tracker_manager.update(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detections=selected_detections,
                )
                if detections is not None:
                    summary.raw_detections_count = len(detections)
                    summary.cat_detections_count = len(detections)
                    summary.acquire_candidate_count = sum(
                        1
                        for detection in detections
                        if detection.confidence >= config.tracking.acquire_confidence_threshold
                    )
                    summary.keep_candidate_count = sum(
                        1
                        for detection in detections
                        if detection.confidence >= config.tracking.keep_confidence_threshold
                    )

            last_summary = summary
            _apply_surface_monitoring(
                summary=summary,
                surface_monitor=surface_monitor,
                frame_shape=frame.shape,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            _log_frame_debug(
                logger=logger,
                summary=summary,
                frame_index=frame_index,
                processed_this_frame=summary.yolo_ran_this_frame,
                enabled=config.logging.per_frame_debug,
            )

            if config.scene_zones.enabled and config.scene_zones.draw_zones:
                overlay.draw_scene_zones(frame, zone_classifier.enabled_zones)

            if summary.tracking_enabled:
                overlay.draw_tracks(
                    frame,
                    summary.visible_tracks,
                    track_location_states=(
                        summary.track_location_states
                        if config.scene_zones.draw_track_locations
                        else None
                    ),
                )
                for track in summary.visible_tracks:
                    coordinate_logger.log_track(track=track, frame_index=frame_index)
            else:
                overlay.draw_detections(frame, summary.detection_overlays)
                if summary.yolo_ran_this_frame:
                    _log_detection_overlays(
                        logger=logger,
                        summary=summary,
                        frame_index=frame_index,
                        log_coordinates=config.logging.log_coordinates,
                    )

            if summary.surface_overlay_message:
                overlay.draw_surface_alert_message(frame, summary.surface_overlay_message)
            overlay.draw_status(
                frame=frame,
                status_text=_build_status_text(summary),
                summary=summary,
                source_status=source.status,
                mode_text=_build_mode_text(config, summary),
            )
            overlay.draw_cat_count(frame, summary.visible_count if summary.tracking_enabled else summary.detection_overlay_count)
            overlay.draw_fps(frame, fps)
            summary.alert_recording_state = alert_recorder.process_frame(
                frame=frame,
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                active_alert_tracks=summary.active_alert_tracks,
            )

            if config.output.save_output:
                if video_writer is None:
                    video_writer = _create_video_writer(
                        output_path=config.output.output_path,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        fps=fps,
                    )
                    if video_writer is None:
                        logger.error("Output writer could not be initialized. Continuing without video saving.")
                if video_writer is not None:
                    video_writer.write(frame)

            if window_enabled:
                try:
                    cv2.imshow(config.output.window_name, frame)
                except cv2.error:
                    logger.exception("OpenCV window update failed. Disabling GUI output.")
                    window_enabled = False
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Shutdown requested by user via 'q'.")
                        break
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 0
    except VideoSourceError as exc:
        logger.error(str(exc))
        return 4
    finally:
        coordinate_logger.close()
        alert_recorder.close()
        surface_monitor.close()
        source.release()
        if video_writer is not None:
            video_writer.release()
        if window_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                logger.debug("OpenCV window cleanup failed.", exc_info=True)


def _process_detection_only_frame(
    detector: YOLODetector,
    frame: np.ndarray,
    frame_index: int,
    timestamp: float,
    detection_cycle_due: bool,
    config: AppConfig,
    selector: TargetSelector,
    last_summary: FrameTrackingSummary | None,
) -> FrameTrackingSummary:
    if not detection_cycle_due and last_summary is not None:
        return replace(
            last_summary,
            frame_index=frame_index,
            yolo_ran_this_frame=False,
            pipeline_state=TrackingPipelineState.SEARCH,
        )

    detections = detector.detect(frame)
    summary = process_detections(
        detections=detections,
        frame_shape=frame.shape,
        frame_index=frame_index,
        timestamp=timestamp,
        config=config,
        selector=selector,
        tracker=None,
    )
    summary.yolo_ran_this_frame = True
    summary.pipeline_state = TrackingPipelineState.SEARCH
    return summary


def process_detections(
    detections: list[Detection],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    frame_index: int,
    timestamp: float,
    config: AppConfig,
    selector: TargetSelector,
    tracker: MultiCatTracker | None,
) -> FrameTrackingSummary:
    raw_detections_count = len(detections)
    cat_detections_count = len(detections)
    acquire_candidate_count = sum(
        1
        for detection in detections
        if detection.confidence >= config.tracking.acquire_confidence_threshold
    )
    keep_candidate_count = sum(
        1
        for detection in detections
        if detection.confidence >= config.tracking.keep_confidence_threshold
    )

    if not config.tracking.tracking_enabled or tracker is None:
        detection_overlays = build_detection_overlays(detections, frame_shape)
        return FrameTrackingSummary(
            frame_index=frame_index,
            frame_width=frame_shape[1],
            frame_height=frame_shape[0],
            tracking_enabled=False,
            detections_count=len(detection_overlays),
            raw_detections_count=raw_detections_count,
            cat_detections_count=cat_detections_count,
            acquire_candidate_count=acquire_candidate_count,
            keep_candidate_count=keep_candidate_count,
            detection_overlays=detection_overlays,
        )

    selected_detections = _select_detections_for_mode(
        detections=detections,
        frame_shape=frame_shape,
        config=config,
        selector=selector,
    )
    summary = tracker.update(
        detections=selected_detections,
        frame_shape=frame_shape,
        frame_index=frame_index,
        timestamp=timestamp,
    )
    summary.raw_detections_count = raw_detections_count
    summary.cat_detections_count = cat_detections_count
    summary.acquire_candidate_count = acquire_candidate_count
    summary.keep_candidate_count = keep_candidate_count
    summary.yolo_ran_this_frame = True
    return summary


def _select_detections_for_mode(
    detections: list[Detection],
    frame_shape: tuple[int, int] | tuple[int, int, int],
    config: AppConfig,
    selector: TargetSelector,
) -> list[Detection]:
    if config.tracking.multi_target_enabled:
        return detections
    selected_target = selector.select_target(detections, frame_shape)
    if selected_target is None:
        return []
    return [selected_target.detection]


def _log_frame_debug(
    logger: logging.Logger,
    summary: FrameTrackingSummary,
    frame_index: int,
    processed_this_frame: bool,
    enabled: bool,
) -> None:
    if not enabled:
        return
    logger.debug(
        "frame_debug frame_index=%d processed=%s pipeline=%s yolo_ran=%s "
        "raw_detections=%d cat_detections=%d after_acquire=%d after_keep=%d "
        "tracker_updates=%d tracker_failures=%d active_tracks=%d confirmed=%d held=%d lost=%d tracking_enabled=%s",
        frame_index,
        processed_this_frame,
        summary.pipeline_state.value,
        summary.yolo_ran_this_frame,
        summary.raw_detections_count,
        summary.cat_detections_count,
        summary.acquire_candidate_count,
        summary.keep_candidate_count,
        summary.tracker_updates_count,
        summary.tracker_failures_count,
        summary.active_tracks_count,
        summary.confirmed_count,
        summary.held_count,
        summary.lost_count,
        summary.tracking_enabled,
    )


def _build_status_text(summary: FrameTrackingSummary) -> str:
    if not summary.tracking_enabled:
        if summary.detection_overlay_count > 0:
            return f"Detected {summary.detection_overlay_count} cat(s)"
        return "Cat not detected"

    if summary.pipeline_state == TrackingPipelineState.REACQUIRE:
        if summary.visible_count > 0:
            return f"Tracking {summary.visible_count} cat(s) | reacquiring"
        return "Reacquiring cat tracks"
    if summary.pipeline_state == TrackingPipelineState.LOST:
        return "Cat lost"
    if summary.visible_count > 0:
        if summary.held_count > 0:
            return f"Tracking {summary.visible_count} cat(s) | {summary.held_count} held"
        return f"Tracking {summary.visible_count} cat(s)"
    if summary.tentative_count > 0:
        return "Confirming new cat tracks"
    if summary.lost_count > 0:
        return "Cats temporarily lost"
    return "Cat not detected"


def _apply_surface_monitoring(
    summary: FrameTrackingSummary,
    surface_monitor: SurfaceMonitor,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    frame_index: int,
    timestamp: float,
) -> None:
    surface_monitor.cleanup_removed_tracks(summary.removed_track_ids)
    result = surface_monitor.update(
        tracks=summary.visible_tracks,
        frame_shape=frame_shape,
        frame_index=frame_index,
        timestamp=timestamp,
    )
    summary.track_location_states = result.track_location_states
    summary.surface_events = result.surface_events
    summary.active_alert_tracks = result.active_alert_tracks
    summary.surface_overlay_message = result.overlay_message


def _build_mode_text(config: AppConfig, summary: FrameTrackingSummary) -> str:
    if not config.tracking.tracking_enabled:
        mode_text = "detection-only"
    elif config.tracking.multi_target_enabled:
        mode_text = "multi-track"
    else:
        mode_text = f"single-track | {config.target_selection_strategy}"

    if config.tracking.tracking_enabled and config.tracking.tracker_only_mode_after_confirm:
        mode_text = f"{mode_text} | detect-then-track | {summary.pipeline_state.value.lower()}"

    if config.source.process_every_n_frames > 1:
        return f"{mode_text} | search 1/{config.source.process_every_n_frames}"
    return mode_text


def _log_detection_overlays(
    logger: logging.Logger,
    summary: FrameTrackingSummary,
    frame_index: int,
    log_coordinates: bool,
) -> None:
    if not log_coordinates:
        return
    for detection in summary.detection_overlays:
        logger.info(
            "frame_index=%d, detection=%d, center=(%d,%d), normalized=(%.3f,%.3f), confidence=%.3f",
            frame_index,
            detection.display_number or 0,
            int(round(detection.center_x)),
            int(round(detection.center_y)),
            detection.normalized_x,
            detection.normalized_y,
            detection.confidence,
        )


def _adjust_detector_threshold_for_runtime_mode(
    config: AppConfig,
    logger: logging.Logger,
    detector_threshold_overridden: bool,
) -> None:
    if detector_threshold_overridden:
        return

    # В detect-then-track режиме не занижаем detector threshold автоматически.
    # Иначе YOLO начинает плодить слишком много слабых детекций и ложных треков.
    if config.tracking.tracking_enabled:
        logger.info(
            "Keeping detector confidence threshold at %.3f for tracking mode.",
            config.detector.confidence_threshold,
        )
        return

    # Только для detection-only можно мягко опустить порог, если очень хочется
    effective_threshold = min(
        config.detector.confidence_threshold,
        config.tracking.keep_confidence_threshold,
    )
    if effective_threshold < config.detector.confidence_threshold:
        logger.info(
            "Lowering detector confidence threshold from %.3f to %.3f for detection-only mode.",
            config.detector.confidence_threshold,
            effective_threshold,
        )
        config.detector.confidence_threshold = effective_threshold


def _maybe_resize_frame(frame: np.ndarray, config: AppConfig) -> np.ndarray:
    if not config.resize.enabled:
        return frame
    return cv2.resize(
        frame,
        (config.resize.width, config.resize.height),
        interpolation=cv2.INTER_LINEAR,
    )


def _should_process_frame(
    frame_index: int,
    process_every_n_frames: int,
    last_summary: FrameTrackingSummary | None,
) -> bool:
    if last_summary is None:
        return True
    return ((frame_index - 1) % process_every_n_frames) == 0


def _initialize_window(config: AppConfig, logger: logging.Logger) -> bool:
    if not config.output.show_window:
        return False
    try:
        cv2.namedWindow(config.output.window_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        logger.exception("Failed to initialize OpenCV window. Continuing in headless mode.")
        return False
    return True


def _create_video_writer(
    output_path: str,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> cv2.VideoWriter | None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    candidates = ["XVID", "MJPG"] if path.suffix.lower() == ".avi" else ["mp4v", "avc1"]
    effective_fps = fps if fps > 0 else 25.0

    for codec in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            effective_fps,
            (frame_width, frame_height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
