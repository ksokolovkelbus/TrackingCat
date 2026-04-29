from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
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
    p = argparse.ArgumentParser(description="Render 6-panel YOLO26s/YOLO26n resolution comparison grid.")
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


def _build_pipeline(model_path: str, imgsz: int, conf: float, logger):
    config = _make_config(model_path, imgsz, conf)
    detector = YOLODetector(config.detector, logger)
    overlay = OverlayRenderer(config.overlay)
    zones = SceneZoneClassifier(config.scene_zones)
    surface = SurfaceMonitor(zones, config.surface_alert, logger, _SilentAudioAlertPlayer(config.surface_alert, logger))
    title = f"{Path(model_path).stem} | imgsz={imgsz} | conf={conf:.2f}"
    return {"config": config, "detector": detector, "overlay": overlay, "zones": zones, "surface": surface, "title": title}


def _resize_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _compose_grid(panels: list[np.ndarray], width: int, height: int) -> np.ndarray:
    resized = [_resize_panel(panel, width, height) for panel in panels]
    sep_v = np.full((height, 4, 3), (40, 40, 40), dtype=np.uint8)
    row1 = np.hstack([resized[0], sep_v, resized[1], sep_v, resized[2]])
    row2 = np.hstack([resized[3], sep_v, resized[4], sep_v, resized[5]])
    sep_h = np.full((4, row1.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    return np.vstack([row1, sep_h, row2])


def main() -> int:
    args = _parse_args()
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logger = setup_logging("INFO")
    video = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Cannot open video: {video}")
        return 2
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    output = Path(args.output) if args.output else Path("recordings/iphone") / f"grid_s_n_resolutions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4"

    specs = []
    for model_path in ["yolo26s.pt", "yolo26n.pt"]:
        for imgsz in args.resolutions:
            specs.append((model_path, imgsz))
    pipelines = [_build_pipeline(model, imgsz, args.conf, logger) for model, imgsz in specs]

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
            if args.seconds > 0 and (source_frame / source_fps) > args.seconds:
                break
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break
            live_fps = processed / max(0.001, time.perf_counter() - started)
            panels = []
            for pipe in pipelines:
                panel, _, _ = _draw_panel(
                    frame,
                    pipe["title"],
                    pipe["detector"],
                    pipe["config"],
                    pipe["overlay"],
                    pipe["zones"],
                    pipe["surface"],
                    source_frame,
                    processed,
                    live_fps,
                )
                panels.append(panel)
            grid = _compose_grid(panels, args.panel_width, args.panel_height)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps / args.stride, (grid.shape[1], grid.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open writer: {output}")
                print(f"Saving: {output}")
            writer.write(grid)
            if args.show:
                cv2.imshow("YOLO26s vs YOLO26n resolution grid", grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if processed % 10 == 0:
                print(f"processed={processed} source_frame={source_frame} render_speed={live_fps:.2f} fps", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for pipe in pipelines:
            pipe["surface"].close()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Done: {processed} frames")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
