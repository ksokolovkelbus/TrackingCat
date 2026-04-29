from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml


def _load_source_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = data.get("source") or {}
    if not isinstance(source, dict):
        return {}
    return source


def _open_capture(camera_index: int, width: int | None, height: int | None, buffer_size: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:
        pass
    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    return capture


def _make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    # mp4v is widely available in OpenCV builds and matches existing project defaults.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a raw camera sample for TrackingCat tuning.")
    parser.add_argument("--config", default="configs/webcam_yolo_track.yaml", help="Config to copy camera quality from.")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0, help="Output file FPS metadata.")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means record until q/Ctrl+C.")
    parser.add_argument("--output-dir", default="recordings")
    parser.add_argument("--name", default=None, help="Optional basename without extension.")
    parser.add_argument("--show", action="store_true", help="Show preview window while recording.")
    parser.add_argument("--buffer-size", type=int, default=None)
    args = parser.parse_args()

    source_defaults = _load_source_defaults(args.config)
    camera_index = args.camera_index
    if camera_index is None:
        camera_index = int(source_defaults.get("camera_index", 0))
    width = args.width if args.width is not None else source_defaults.get("camera_width")
    height = args.height if args.height is not None else source_defaults.get("camera_height")
    buffer_size = args.buffer_size
    if buffer_size is None:
        buffer_size = int(source_defaults.get("buffer_size", 1))
    width = int(width) if width is not None else None
    height = int(height) if height is not None else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = args.name or f"cat_camera_sample_{timestamp}"
    video_path = output_dir / f"{basename}.mp4"
    meta_path = output_dir / f"{basename}.json"

    capture = _open_capture(camera_index, width, height, buffer_size)
    actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    writer = _make_writer(video_path, args.fps, actual_width, actual_height)

    duration_label = "until stopped" if args.seconds <= 0 else f"{args.seconds:.1f}s"
    print(f"Recording camera {camera_index}: requested={width}x{height}, actual={actual_width}x{actual_height}, capture_fps={capture_fps:.2f}")
    print(f"Config: {args.config}")
    print(f"Duration: {duration_label}")
    print(f"Output: {video_path}")
    print("Stop with q in preview window or Ctrl+C.")

    started = time.perf_counter()
    frames = 0
    stopped_by = "manual"
    try:
        while True:
            elapsed = time.perf_counter() - started
            if args.seconds > 0 and elapsed >= args.seconds:
                stopped_by = "duration"
                break
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            writer.write(frame)
            frames += 1
            if args.show:
                preview = frame.copy()
                text = f"REC {frames} frames {elapsed:.1f}s"
                if args.seconds > 0:
                    text += f"/{args.seconds:.1f}s"
                cv2.putText(
                    preview,
                    text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("TrackingCat sample recorder", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by = "q"
                    break
    except KeyboardInterrupt:
        stopped_by = "ctrl_c"
    finally:
        duration = max(0.001, time.perf_counter() - started)
        writer.release()
        capture.release()
        if args.show:
            cv2.destroyAllWindows()

    metadata = {
        "video_path": str(video_path),
        "config": args.config,
        "camera_index": camera_index,
        "requested_width": width,
        "requested_height": height,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "output_fps": args.fps,
        "capture_reported_fps": capture_fps,
        "recorded_frames": frames,
        "duration_seconds": duration,
        "effective_record_fps": frames / duration,
        "stopped_by": stopped_by,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done: {frames} frames in {duration:.1f}s ({frames / duration:.1f} FPS)")
    print(f"Video: {video_path}")
    print(f"Meta:  {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
