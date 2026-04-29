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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render one YOLO model at several resolutions separately, then stitch horizontally.")
    p.add_argument("video")
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--label", default="YOLO26m")
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


def make_config(model: str, imgsz: int, conf: float):
    c = load_config("configs/iphone_fast20_bytetrack.yaml")
    c.detector.model_path = model
    c.detector.imgsz = imgsz
    c.detector.confidence_threshold = conf
    c.detector.iou_threshold = 0.6
    c.detector.max_frame_area_ratio = 0.75
    c.tracking.backend = "bytetrack"
    c.tracking.yolo_tracker = "configs/bytetrack_cats_fast20.yaml"
    c.tracking.acquire_confidence_threshold = conf
    c.tracking.keep_confidence_threshold = max(0.03, conf * 0.7)
    c.output.show_window = False
    c.output.save_output = False
    c.surface_alert.enabled = False
    c.alert_recording.enabled = False
    c.overlay.debug_overlay = False
    c.overlay.show_debug_counters = False
    c.overlay.show_cat_count = True
    return c


def source_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    cap.release()
    return fps


def resize(frame, w, h):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def render_one(args, video: Path, fps: float, imgsz: int, out: Path, logger) -> int:
    config = make_config(args.model, imgsz, args.conf)
    detector = YOLODetector(config.detector, logger)
    overlay = OverlayRenderer(config.overlay)
    zones = SceneZoneClassifier(config.scene_zones)
    surface = SurfaceMonitor(zones, config.surface_alert, logger, _SilentAudioAlertPlayer(config.surface_alert, logger))
    title = f"{args.label} | imgsz={imgsz} | conf={args.conf:.2f}"
    cap = cv2.VideoCapture(str(video))
    writer = None
    processed = 0
    source_frame = 0
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
            speed = processed / max(0.001, time.perf_counter() - started)
            panel, _, _ = _draw_panel(frame, title, detector, config, overlay, zones, surface, source_frame, processed, speed)
            panel = resize(panel, args.panel_width, args.panel_height)
            if writer is None:
                out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (panel.shape[1], panel.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open writer: {out}")
            writer.write(panel)
            if processed % 50 == 0:
                print(f"{title}: rendered={processed} speed={speed:.2f} fps", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        surface.close()
    print(f"{title}: done rendered={processed} -> {out}")
    return processed


def stitch(parts: list[Path], output: Path, fps: float, stride: int, show: bool) -> int:
    caps = [cv2.VideoCapture(str(p)) for p in parts]
    writer = None
    frames = 0
    try:
        while True:
            read = [c.read() for c in caps]
            if not all(ok and frame is not None for ok, frame in read):
                break
            panels = [frame for _ok, frame in read]
            h = panels[0].shape[0]
            sep = np.full((h, 4, 3), (40, 40, 40), dtype=np.uint8)
            row = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (row.shape[1], row.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output writer: {output}")
                print(f"Saving stitched row: {output}")
            writer.write(row)
            frames += 1
            if show:
                cv2.imshow("model resolution row", row)
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
    args = parse_args()
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logger = setup_logging("INFO")
    video = Path(args.video).expanduser()
    fps = source_fps(video)
    stem = Path(args.model).stem.replace(".", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else Path("recordings/iphone") / f"{stem}_resolutions_{stamp}.mp4"
    tmp = output.parent / f".{output.stem}_parts"
    parts = []
    for i, imgsz in enumerate(args.resolutions, start=1):
        part = tmp / f"{stem}_{imgsz}.mp4"
        print(f"Pass {i}/{len(args.resolutions)+1}: render {args.label} imgsz={imgsz}")
        render_one(args, video, fps, imgsz, part, logger)
        parts.append(part)
    print(f"Pass {len(args.resolutions)+1}/{len(args.resolutions)+1}: stitch")
    frames = stitch(parts, output, fps, args.stride, args.show)
    print(f"Done: {frames} stitched frames")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
