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

from app.config import load_config
from app.detector import YOLODetector
from app.logger_setup import setup_logging
from app.overlay import OverlayRenderer
from app.surface_monitor import SurfaceMonitor
from app.audio_alert import AudioAlertPlayer
from app.zones import SceneZoneClassifier
from tools.compare_tracking_modes import _draw_panel, _compose


class _SilentAudioAlertPlayer(AudioAlertPlayer):
    def play(self) -> None:  # type: ignore[override]
        return None

    def stop(self) -> None:  # type: ignore[override]
        return None

    def close(self) -> None:  # type: ignore[override]
        return None


def _prepare_config(path: str, device: str | None):
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline two-pass TrackingCat comparison: render each config separately, then stitch side-by-side.")
    parser.add_argument("video")
    parser.add_argument("--left-config", required=True)
    parser.add_argument("--right-config", required=True)
    parser.add_argument("--left-title", default="Left")
    parser.add_argument("--right-title", default="Right")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true", help="Preview stitched output while final stitching pass runs.")
    return parser.parse_args()


def _source_info(video: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, width, height


def _render_one(
    *,
    video: Path,
    config_path: str,
    title: str,
    output_path: Path,
    fps: float,
    args: argparse.Namespace,
    logger,
) -> int:
    config = _prepare_config(config_path, args.device)
    detector = YOLODetector(config.detector, logger)
    overlay = OverlayRenderer(config.overlay)
    zones = SceneZoneClassifier(config.scene_zones)
    surface = SurfaceMonitor(zones, config.surface_alert, logger, _SilentAudioAlertPlayer(config.surface_alert, logger))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    writer = None
    source_frame = 0
    processed = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            source_frame += 1
            if (source_frame - 1) % args.stride != 0:
                continue
            if args.seconds > 0 and (source_frame / fps) > args.seconds:
                break
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break
            live_fps = processed / max(0.001, time.perf_counter() - started)
            panel, _, _ = _draw_panel(frame, title, detector, config, overlay, zones, surface, source_frame, processed, live_fps)
            if args.scale != 1.0:
                w = max(1, int(round(panel.shape[1] * args.scale)))
                h = max(1, int(round(panel.shape[0] * args.scale)))
                panel = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA if args.scale < 1 else cv2.INTER_LINEAR)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (panel.shape[1], panel.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open writer: {output_path}")
            writer.write(panel)
            if processed % 50 == 0:
                print(f"{title}: rendered={processed} source_frame={source_frame} speed={live_fps:.2f} fps", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        surface.close()
    print(f"{title}: done rendered={processed} -> {output_path}")
    return processed


def _stitch(left_path: Path, right_path: Path, output_path: Path, fps: float, stride: int, show: bool) -> int:
    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Cannot open intermediate videos for stitching")
    writer = None
    frames = 0
    try:
        while True:
            ok_l, frame_l = left.read()
            ok_r, frame_r = right.read()
            if not ok_l or not ok_r or frame_l is None or frame_r is None:
                break
            composed = _compose(frame_l, frame_r, 1.0)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (composed.shape[1], composed.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output writer: {output_path}")
            writer.write(composed)
            frames += 1
            if show:
                cv2.imshow("TrackingCat offline comparison", composed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        left.release()
        right.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return frames


def main() -> int:
    args = _parse_args()
    if args.stride < 1:
        print("--stride must be >= 1")
        return 2
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logger = setup_logging("INFO")
    video = Path(args.video).expanduser()
    fps, _, _ = _source_info(video)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else Path("recordings/iphone") / f"offline_compare_{stamp}.mp4"
    tmp_dir = output.parent / f".{output.stem}_parts"
    left_tmp = tmp_dir / "left.mp4"
    right_tmp = tmp_dir / "right.mp4"

    print("Pass 1/3: render left config")
    _render_one(video=video, config_path=args.left_config, title=args.left_title, output_path=left_tmp, fps=fps, args=args, logger=logger)
    print("Pass 2/3: render right config")
    _render_one(video=video, config_path=args.right_config, title=args.right_title, output_path=right_tmp, fps=fps, args=args, logger=logger)
    print("Pass 3/3: stitch side-by-side")
    frames = _stitch(left_tmp, right_tmp, output, fps, args.stride, args.show)
    print(f"Done: {frames} stitched frames")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
