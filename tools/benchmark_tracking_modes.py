from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from app.config import load_config
from app.detector import YOLODetector
from app.logger_setup import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TrackingCat configs on the same video frames.")
    parser.add_argument("video")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--sample-every", type=int, default=10, help="Use every Nth source frame.")
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def load_frames(video: Path, sample_every: int, max_samples: int):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    frames = []
    source_index = 0
    while len(frames) < max_samples:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        source_index += 1
        if source_index % sample_every == 0:
            frames.append(frame)
    capture.release()
    return frames


def main() -> int:
    args = parse_args()
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logger = setup_logging("INFO")
    names = args.names or args.configs
    if len(names) != len(args.configs):
        print("--names count must match --configs count")
        return 2
    frames = load_frames(Path(args.video).expanduser(), args.sample_every, args.max_samples)
    if not frames:
        print("No frames loaded")
        return 3
    print(f"Loaded {len(frames)} sampled frames from {args.video}")
    print("name\tmodel\timgsz\tconf\tavg_ms\tfps\tavg_cats\tzero_frames")
    for name, config_path in zip(names, args.configs):
        config = load_config(config_path)
        detector = YOLODetector(config.detector, logger)
        for frame in frames[: args.warmup]:
            detector.track(frame, tracker=config.tracking.yolo_tracker)
        counts = []
        started = time.perf_counter()
        for frame in frames:
            detections = detector.track(frame, tracker=config.tracking.yolo_tracker)
            counts.append(len(detections))
        elapsed = time.perf_counter() - started
        avg_ms = elapsed / len(frames) * 1000.0
        fps = len(frames) / elapsed
        avg_cats = sum(counts) / len(counts)
        zero_frames = sum(1 for count in counts if count == 0)
        print(f"{name}\t{config.detector.model_path}\t{config.detector.imgsz}\t{config.detector.confidence_threshold:.3f}\t{avg_ms:.1f}\t{fps:.2f}\t{avg_cats:.2f}\t{zero_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
