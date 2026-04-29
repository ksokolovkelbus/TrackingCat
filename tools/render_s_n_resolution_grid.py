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
from app.overlay import OverlayRenderer
from app.surface_monitor import SurfaceMonitor
from app.audio_alert import AudioAlertPlayer
from app.zones import SceneZoneClassifier
from tools.compare_tracking_modes import _draw_panel


class _SilentAudioAlertPlayer(AudioAlertPlayer):
    def play(self) -> None: return None  # type: ignore[override]
    def stop(self) -> None: return None  # type: ignore[override]
    def close(self) -> None: return None  # type: ignore[override]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline separate-pass 6-panel YOLO26s/YOLO26n resolution comparison grid.")
    p.add_argument("video")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--resolutions", nargs="+", type=int, default=[640, 384, 256])
    p.add_argument("--output", default=None)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--panel-width", type=int, default=426)
    p.add_argument("--panel-height", type=int, default=320)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _make_config(model_path: str, imgsz: int, conf: float):
    config = load_config("configs/iphone_fast20_bytetrack.yaml")
    config.detector.model_path = model_path
    config.detector.imgsz = imgsz
    config.detector.confidence_threshold = conf
    config.detector.iou_threshold = 0.6
    config.detector.max_frame_area_ratio = 0.75
    config.tracking.backend = "bytetrack"
    config.tracking.yolo_tracker = "configs/bytetrack_cats_fast20.yaml"
    config.tracking.acquire_confidence_threshold = conf
    config.tracking.keep_confidence_threshold = max(0.03, conf * 0.7)
    config.output.show_window = False
    config.output.save_output = False
    config.surface_alert.enabled = False
    config.alert_recording.enabled = False
    config.overlay.debug_overlay = False
    config.overlay.show_debug_counters = False
    config.overlay.show_cat_count = True
    return config


def _source_info(video: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, width, height


def _resize_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _render_one(
    *,
    video: Path,
    model_path: str,
    imgsz: int,
    conf: float,
    title: str,
    output_path: Path,
    fps: float,
    args: argparse.Namespace,
    logger,
) -> int:
    config = _make_config(model_path, imgsz, conf)
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
            render_fps = processed / max(0.001, time.perf_counter() - started)
            panel, _, _ = _draw_panel(frame, title, detector, config, overlay, zones, surface, source_frame, processed, render_fps)
            panel = _resize_panel(panel, args.panel_width, args.panel_height)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (panel.shape[1], panel.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open writer: {output_path}")
            writer.write(panel)
            if processed % 50 == 0:
                print(f"{title}: rendered={processed} source_frame={source_frame} speed={render_fps:.2f} fps", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        surface.close()
    print(f"{title}: done rendered={processed} -> {output_path}")
    return processed


def _stitch(parts: list[Path], output: Path, fps: float, stride: int, show: bool) -> int:
    caps = [cv2.VideoCapture(str(p)) for p in parts]
    if not all(c.isOpened() for c in caps):
        raise RuntimeError("Cannot open all intermediate videos")
    writer = None
    frames = 0
    try:
        while True:
            read = [c.read() for c in caps]
            if not all(ok and frame is not None for ok, frame in read):
                break
            panels = [frame for _ok, frame in read]
            h, w = panels[0].shape[:2]
            sep_v = np.full((h, 4, 3), (40, 40, 40), dtype=np.uint8)
            row_s = np.hstack([panels[0], sep_v, panels[1], sep_v, panels[2]])
            row_n = np.hstack([panels[3], sep_v, panels[4], sep_v, panels[5]])
            sep_h = np.full((4, row_s.shape[1], 3), (40, 40, 40), dtype=np.uint8)
            grid = np.vstack([row_s, sep_h, row_n])
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (grid.shape[1], grid.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output writer: {output}")
                print(f"Saving stitched grid: {output}")
            writer.write(grid)
            frames += 1
            if show:
                cv2.imshow("YOLO26s vs YOLO26n resolution grid", grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for c in caps:
            c.release()
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
    output = Path(args.output) if args.output else Path("recordings/iphone") / f"grid_s_n_resolutions_{stamp}.mp4"
    tmp_dir = output.parent / f".{output.stem}_parts"
    specs = []
    for row_name, model_path in [("S", "yolo26s.pt"), ("N", "yolo26n.pt")]:
        for imgsz in args.resolutions:
            title = f"YOLO26{row_name.lower()} | imgsz={imgsz} | conf={args.conf:.2f}"
            part = tmp_dir / f"{row_name}_{imgsz}.mp4"
            specs.append((model_path, imgsz, title, part))

    for index, (model_path, imgsz, title, part) in enumerate(specs, start=1):
        print(f"Pass {index}/7: render {title}")
        _render_one(video=video, model_path=model_path, imgsz=imgsz, conf=args.conf, title=title, output_path=part, fps=fps, args=args, logger=logger)
    print("Pass 7/7: stitch 6 rendered videos")
    frames = _stitch([part for *_rest, part in specs], output, fps, args.stride, args.show)
    print(f"Done: {frames} stitched frames")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
