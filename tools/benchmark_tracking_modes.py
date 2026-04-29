from __future__ import annotations

import argparse
import logging
import statistics
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
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames not included in timing.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat timed pass N times on already loaded models.")
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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


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
    warmup_frames = frames[: min(args.warmup, len(frames))]
    print(f"Loaded {len(frames)} sampled frames from {args.video}")
    print(f"Warmup frames per config: {len(warmup_frames)}; timed repeats: {args.repeats}")
    print("name\tmodel\timgsz\tconf\trepeat\tavg_ms\tmedian_ms\tp95_ms\tfps\tavg_cats\tzero_frames")
    for name, config_path in zip(names, args.configs):
        config = load_config(config_path)
        load_started = time.perf_counter()
        detector = YOLODetector(config.detector, logger)
        load_ms = (time.perf_counter() - load_started) * 1000.0
        for frame in warmup_frames:
            detector.track(frame, tracker=config.tracking.yolo_tracker)
        print(f"# loaded {name} in {load_ms:.1f} ms; load time excluded from inference rows")
        for repeat in range(1, args.repeats + 1):
            times_ms: list[float] = []
            counts: list[int] = []
            for frame in frames:
                started = time.perf_counter()
                detections = detector.track(frame, tracker=config.tracking.yolo_tracker)
                times_ms.append((time.perf_counter() - started) * 1000.0)
                counts.append(len(detections))
            avg_ms = statistics.fmean(times_ms)
            median_ms = statistics.median(times_ms)
            p95_ms = percentile(times_ms, 95)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            avg_cats = statistics.fmean(counts)
            zero_frames = sum(1 for count in counts if count == 0)
            print(
                f"{name}\t{config.detector.model_path}\t{config.detector.imgsz}\t{config.detector.confidence_threshold:.3f}\t"
                f"{repeat}\t{avg_ms:.1f}\t{median_ms:.1f}\t{p95_ms:.1f}\t{fps:.2f}\t{avg_cats:.2f}\t{zero_frames}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
