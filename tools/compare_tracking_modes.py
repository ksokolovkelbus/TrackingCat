from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.config import load_config
from app.detector import YOLODetector
from app.logger_setup import setup_logging
from app.main import _apply_surface_monitoring, _build_mode_text, _build_status_text, _build_yolo_native_tracks
from app.models import AppConfig, FrameTrackingSummary, TrackingPipelineState
from app.overlay import OverlayRenderer
from app.surface_monitor import SurfaceMonitor
from app.audio_alert import AudioAlertPlayer
from app.zones import SceneZoneClassifier


class _SilentAudioAlertPlayer(AudioAlertPlayer):
    def play(self) -> None:  # type: ignore[override]
        return None

    def stop(self) -> None:  # type: ignore[override]
        return None

    def close(self) -> None:  # type: ignore[override]
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two TrackingCat YOLO tracker configs side-by-side on the same video.")
    parser.add_argument("video", help="Input video file, e.g. recordings/iphone/cat_camera_sample_....mp4")
    parser.add_argument("--left-config", default="configs/iphone_yolo_track.yaml")
    parser.add_argument("--right-config", default="configs/iphone_botsort_track.yaml")
    parser.add_argument("--left-title", default="iPhone + ByteTrack")
    parser.add_argument("--right-title", default="iPhone + BoT-SORT")
    parser.add_argument("--output", default=None, help="Output comparison MP4. Default: recordings/iphone/compare_<timestamp>.mp4")
    parser.add_argument("--no-save", action="store_true", help="Do not save comparison video.")
    parser.add_argument("--no-window", action="store_true", help="Do not show live comparison window.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Limit processing duration in source-video seconds; 0 means full video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit processed frames; 0 means no frame limit.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth source frame.")
    parser.add_argument("--scale", type=float, default=0.75, help="Scale each panel before composing the side-by-side window/video.")
    parser.add_argument("--device", default=None, help="Override detector device, e.g. cpu or cuda:0.")
    parser.add_argument("--window-name", default="TrackingCat compare: ByteTrack vs BoT-SORT")
    parser.add_argument("--output-fps", type=float, default=0.0, help="Output FPS metadata; 0 uses source FPS/stride.")
    return parser.parse_args()


def _prepare_config(path: str, device: str | None) -> AppConfig:
    config = load_config(path)
    if device:
        config.detector.device = device
    config.output.show_window = False
    config.output.save_output = False
    config.surface_alert.enabled = False
    config.alert_recording.enabled = False
    config.overlay.debug_overlay = False
    config.overlay.show_debug_counters = False
    config.overlay.show_cat_count = True
    return config


def _process_frame(detector: YOLODetector, config: AppConfig, frame: np.ndarray, frame_index: int) -> FrameTrackingSummary:
    detections = detector.track(frame, tracker=config.tracking.yolo_tracker)
    tracks = _build_yolo_native_tracks(detections=detections, frame_shape=frame.shape, frame_index=frame_index)
    return FrameTrackingSummary(
        frame_index=frame_index,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        tracking_enabled=True,
        detections_count=len(detections),
        pipeline_state=TrackingPipelineState.TRACK_ONLY,
        yolo_ran_this_frame=True,
        raw_detections_count=len(detections),
        cat_detections_count=len(detections),
        acquire_candidate_count=sum(1 for detection in detections if detection.confidence >= config.tracking.acquire_confidence_threshold),
        keep_candidate_count=sum(1 for detection in detections if detection.confidence >= config.tracking.keep_confidence_threshold),
        visible_tracks=tracks,
    )


def _draw_panel(
    frame: np.ndarray,
    title: str,
    detector: YOLODetector,
    config: AppConfig,
    overlay: OverlayRenderer,
    zone_classifier: SceneZoneClassifier,
    surface_monitor: SurfaceMonitor,
    source_frame_index: int,
    processed_index: int,
    fps: float,
) -> tuple[np.ndarray, FrameTrackingSummary, float]:
    panel = frame.copy()
    started = time.perf_counter()
    summary = _process_frame(detector=detector, config=config, frame=panel, frame_index=processed_index)
    inference_ms = (time.perf_counter() - started) * 1000.0
    _apply_surface_monitoring(
        summary=summary,
        surface_monitor=surface_monitor,
        frame_shape=panel.shape,
        frame_index=processed_index,
        timestamp=time.time(),
    )
    if config.scene_zones.enabled and config.scene_zones.draw_zones:
        overlay.draw_scene_zones(panel, zone_classifier.enabled_zones)
    overlay.draw_tracks(
        panel,
        summary.visible_tracks,
        track_location_states=summary.track_location_states if config.scene_zones.draw_track_locations else None,
    )
    overlay.draw_status(
        frame=panel,
        status_text=_build_status_text(summary),
        summary=summary,
        source_status="file",
        mode_text=_build_mode_text(config, summary),
    )
    overlay.draw_cat_count(panel, summary.visible_count)
    overlay.draw_fps(panel, fps)
    _draw_title(panel, title, source_frame_index, processed_index, summary.visible_count, inference_ms)
    return panel, summary, inference_ms


def _draw_title(frame: np.ndarray, title: str, source_frame_index: int, processed_index: int, cats: int, inference_ms: float) -> None:
    text = f"{title} | cats={cats} | frame={source_frame_index} | processed={processed_index} | yolo={inference_ms:.0f}ms"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 120), 2, cv2.LINE_AA)


def _resize_panel(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    width = max(1, int(round(frame.shape[1] * scale)))
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def _compose(left: np.ndarray, right: np.ndarray, scale: float) -> np.ndarray:
    left = _resize_panel(left, scale)
    right = _resize_panel(right, scale)
    height = max(left.shape[0], right.shape[0])
    if left.shape[0] != height:
        left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if right.shape[0] != height:
        right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    separator = np.zeros((height, 4, 3), dtype=np.uint8)
    separator[:, :] = (40, 40, 40)
    return np.hstack([left, separator, right])


def main() -> int:
    args = _parse_args()
    logger = setup_logging("INFO")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 2
    if args.stride < 1:
        print("--stride must be >= 1")
        return 2
    if args.scale <= 0:
        print("--scale must be > 0")
        return 2

    left_config = _prepare_config(args.left_config, args.device)
    right_config = _prepare_config(args.right_config, args.device)
    left_detector = YOLODetector(left_config.detector, logger)
    right_detector = YOLODetector(right_config.detector, logger)
    left_overlay = OverlayRenderer(left_config.overlay)
    right_overlay = OverlayRenderer(right_config.overlay)
    left_zones = SceneZoneClassifier(left_config.scene_zones)
    right_zones = SceneZoneClassifier(right_config.scene_zones)
    left_surface = SurfaceMonitor(left_zones, left_config.surface_alert, logger, _SilentAudioAlertPlayer(left_config.surface_alert, logger))
    right_surface = SurfaceMonitor(right_zones, right_config.surface_alert, logger, _SilentAudioAlertPlayer(right_config.surface_alert, logger))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Cannot open video: {video_path}")
        return 3

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = args.output_fps if args.output_fps > 0 else (source_fps / args.stride if source_fps > 0 else 15.0)
    output_path = Path(args.output) if args.output else Path("recordings/iphone") / f"compare_bytetrack_botsort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    writer: cv2.VideoWriter | None = None
    processed = 0
    source_frame = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_frame += 1
            if (source_frame - 1) % args.stride != 0:
                continue
            if args.seconds > 0 and source_fps > 0 and (source_frame / source_fps) > args.seconds:
                break
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break

            live_fps = processed / max(0.001, time.perf_counter() - started)
            left_panel, _, _ = _draw_panel(frame, args.left_title, left_detector, left_config, left_overlay, left_zones, left_surface, source_frame, processed, live_fps)
            right_panel, _, _ = _draw_panel(frame, args.right_title, right_detector, right_config, right_overlay, right_zones, right_surface, source_frame, processed, live_fps)
            composed = _compose(left_panel, right_panel, args.scale)

            if not args.no_save:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (composed.shape[1], composed.shape[0]))
                    if not writer.isOpened():
                        print(f"Cannot open output writer: {output_path}")
                        writer = None
                    else:
                        print(f"Saving comparison video: {output_path}")
                if writer is not None:
                    writer.write(composed)

            if not args.no_window:
                cv2.imshow(args.window_name, composed)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if processed % 25 == 0:
                print(f"processed={processed} source_frame={source_frame} live_fps={live_fps:.2f}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        left_surface.close()
        right_surface.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    print(f"Done. processed={processed}, source_frames={source_frame}")
    if not args.no_save:
        print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
