from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

LOG_FILE_PATH = Path.home() / "PycharmProjects" / "TrackingCatWIFI" / "log.txt"

from app.models import AppConfig, Track
from app.utils import safe_float_text


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("tracking_cat")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Log started: %s", LOG_FILE_PATH)
    return logger


class TrackCoordinateLogger:
    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self._logger = logger
        self._enabled = config.logging.log_coordinates
        self._log_path = (
            Path(config.logging.coordinates_log_path)
            if config.logging.coordinates_log_path
            else None
        )
        self._log_format = config.logging.coordinates_log_format
        self._stream: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None

        if self._enabled and self._log_path:
            self._prepare_output()

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None

    def log_track(self, track: Track, frame_index: int) -> None:
        if not self._enabled:
            return

        message = (
            f"track_id={track.track_id}, "
            f"center=({int(round(track.center_x))},{int(round(track.center_y))}), "
            f"normalized=({safe_float_text(track.normalized_x)},"
            f"{safe_float_text(track.normalized_y)}), "
            f"confidence={track.confidence:.3f}"
        )
        self._logger.info(message)

        if not self._stream:
            return

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_index": frame_index,
            "track_id": track.track_id,
            "state": track.state.value,
            "center_x": int(round(track.center_x)),
            "center_y": int(round(track.center_y)),
            "normalized_x": round(track.normalized_x, 6),
            "normalized_y": round(track.normalized_y, 6),
            "confidence": round(track.confidence, 6),
        }
        if self._log_format == "csv":
            if not self._csv_writer:
                raise RuntimeError("CSV writer is not initialized.")
            self._csv_writer.writerow(record)
            self._stream.flush()
            return

        self._stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._stream.flush()

    def _prepare_output(self) -> None:
        if not self._log_path:
            return

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._log_path.open("a", encoding="utf-8", newline="")

        if self._log_format == "csv":
            fieldnames = [
                "timestamp_utc",
                "frame_index",
                "track_id",
                "state",
                "center_x",
                "center_y",
                "normalized_x",
                "normalized_y",
                "confidence",
            ]
            self._csv_writer = csv.DictWriter(self._stream, fieldnames=fieldnames)
            if self._stream.tell() == 0:
                self._csv_writer.writeheader()


TargetCoordinateLogger = TrackCoordinateLogger
