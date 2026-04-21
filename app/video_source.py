from __future__ import annotations

import logging
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import cv2
import numpy as np

from app.browser_camera import BrowserCameraIngestServer
from app.models import SourceConfig


class VideoSourceError(RuntimeError):
    """Raised when video source cannot be opened or recovered."""


class VideoSource:
    def __init__(self, config: SourceConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._capture: cv2.VideoCapture | None = None
        self._consecutive_failures = 0
        self._snapshot_opened = False
        self._pending_frame: np.ndarray | None = None
        self._snapshot_counter = 0
        self._browser_server: BrowserCameraIngestServer | None = None
        self._browser_last_frame_id = 0
        self.status = "disconnected"

    def open(self) -> None:
        if self._config.source_type == "http_snapshot":
            frame = self._fetch_snapshot_frame()
            if frame is None:
                raise VideoSourceError(
                    f"Failed to open source '{self._config.source_type}': {self._safe_source_repr()}"
                )
            self._snapshot_opened = True
            self._pending_frame = frame
            self.status = "connected"
            self._consecutive_failures = 0
            self._log_snapshot_configuration(frame)
            return

        if self._config.source_type == "browser_upload":
            self._browser_server = BrowserCameraIngestServer(config=self._config, logger=self._logger)
            try:
                self._browser_server.start()
            except OSError as exc:
                raise VideoSourceError(
                    f"Failed to start browser camera ingest server on {self._config.browser_camera_host}:{self._config.browser_camera_port}: {exc}"
                ) from exc
            self._snapshot_opened = True
            self._browser_last_frame_id = 0
            self.status = "waiting_for_browser"
            self._consecutive_failures = 0
            self._logger.info("Open %s from the device browser and press Start camera.", self._browser_server.public_url)
            return

        self._capture = self._create_capture()
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise VideoSourceError(
                f"Failed to open source '{self._config.source_type}': {self._safe_source_repr()}"
            )
        self.status = "connected"
        self._consecutive_failures = 0
        self._log_capture_configuration()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._config.source_type == "http_snapshot":
            return self._read_http_snapshot()
        if self._config.source_type == "browser_upload":
            return self._read_browser_upload()

        if not self.is_opened():
            if self._config.source_type == "file":
                self.status = "ended"
                return False, None
            recovered = self._attempt_reconnect()
            if not recovered:
                return False, None

        assert self._capture is not None
        ok, frame = self._capture.read()
        if ok and frame is not None and frame.size > 0:
            self._consecutive_failures = 0
            self.status = "connected"
            return True, frame

        self._consecutive_failures += 1
        if self._config.source_type == "file":
            self.status = "ended"
            return False, None

        if self._consecutive_failures < self._config.read_fail_threshold:
            self.status = "degraded"
            return False, None

        self._logger.warning(
            "Read failure threshold reached for source '%s'. Attempting reconnect.",
            self._safe_source_repr(),
        )
        if self._attempt_reconnect():
            assert self._capture is not None
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.size > 0:
                self._consecutive_failures = 0
                self.status = "connected"
                return True, frame

        self.status = "disconnected"
        return False, None

    def is_opened(self) -> bool:
        if self._config.source_type in {"http_snapshot", "browser_upload"}:
            return self._snapshot_opened
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._browser_server is not None:
            self._browser_server.stop()
            self._browser_server = None
        self._snapshot_opened = False
        self._pending_frame = None
        self._browser_last_frame_id = 0
        self.status = "released"

    def _create_capture(self) -> cv2.VideoCapture:
        source = self._config.resolved_source()
        capture = cv2.VideoCapture(source)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self._config.buffer_size)
        except Exception:
            self._logger.debug("CAP_PROP_BUFFERSIZE is not supported by this backend.")
        if self._config.camera_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.camera_width))
        if self._config.camera_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.camera_height))
        return capture

    def _attempt_reconnect(self) -> bool:
        for attempt in range(1, self._config.max_reconnect_attempts + 1):
            self.status = "reconnecting"
            self._logger.warning(
                "Reconnect attempt %d/%d for source '%s'.",
                attempt,
                self._config.max_reconnect_attempts,
                self._safe_source_repr(),
            )
            self.release()
            time.sleep(self._config.reconnect_delay_seconds)
            try:
                self.open()
                return True
            except VideoSourceError:
                continue
        self.status = "disconnected"
        return False

    def _safe_source_repr(self) -> str:
        if self._config.source_type == "webcam":
            return str(self._config.camera_index)
        if self._config.source_type == "file":
            return self._config.source_path or "<missing file path>"
        if self._config.source_type == "browser_upload":
            return f"http://{self._config.browser_public_host}:{self._config.browser_camera_port}/"
        return self._config.stream_url or "<missing stream url>"

    def _log_capture_configuration(self) -> None:
        if self._capture is None:
            return
        width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._logger.info(
            "Video source opened: source=%s, requested_resolution=%sx%s, actual_resolution=%sx%s, process_every_n_frames=%d",
            self._safe_source_repr(),
            self._config.camera_width if self._config.camera_width is not None else "auto",
            self._config.camera_height if self._config.camera_height is not None else "auto",
            width,
            height,
            self._config.process_every_n_frames,
        )

    def _log_snapshot_configuration(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self._logger.info(
            "Snapshot source opened: source=%s, actual_resolution=%sx%s, process_every_n_frames=%d, timeout=%.2fs",
            self._safe_source_repr(),
            width,
            height,
            self._config.process_every_n_frames,
            self._config.snapshot_timeout_seconds,
        )

    def _read_browser_upload(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_opened() or self._browser_server is None:
            recovered = self._attempt_reconnect()
            if not recovered:
                return False, None

        assert self._browser_server is not None
        frame_id, frame, frame_ts = self._browser_server.get_latest_frame()
        if frame is None:
            self.status = "waiting_for_browser"
            return False, None

        frame_age = time.time() - frame_ts
        if frame_age > self._config.browser_frame_timeout_seconds:
            self.status = "browser_stale"
            return False, None

        if frame_id == self._browser_last_frame_id:
            self.status = "connected"
            return False, None

        self._browser_last_frame_id = frame_id
        self._consecutive_failures = 0
        self.status = "connected"
        return True, frame

    def _read_http_snapshot(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_opened():
            recovered = self._attempt_reconnect()
            if not recovered:
                return False, None

        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            self._consecutive_failures = 0
            self.status = "connected"
            return True, frame

        frame = self._fetch_snapshot_frame()
        if frame is not None:
            self._consecutive_failures = 0
            self.status = "connected"
            return True, frame

        self._consecutive_failures += 1
        if self._consecutive_failures < self._config.read_fail_threshold:
            self.status = "degraded"
            return False, None

        self._logger.warning(
            "Snapshot read failure threshold reached for source '%s'. Attempting reconnect.",
            self._safe_source_repr(),
        )
        if self._attempt_reconnect():
            if self._pending_frame is not None:
                frame = self._pending_frame
                self._pending_frame = None
                self._consecutive_failures = 0
                self.status = "connected"
                return True, frame
            frame = self._fetch_snapshot_frame()
            if frame is not None:
                self._consecutive_failures = 0
                self.status = "connected"
                return True, frame

        self.status = "disconnected"
        return False, None

    def _fetch_snapshot_frame(self) -> np.ndarray | None:
        url = self._build_snapshot_url()
        request = urllib_request.Request(url, headers={"Cache-Control": "no-cache"})
        try:
            with urllib_request.urlopen(request, timeout=self._config.snapshot_timeout_seconds) as response:
                payload = response.read()
        except (urllib_error.URLError, TimeoutError, ValueError, OSError) as exc:
            self._logger.debug("Snapshot fetch failed for %s: %s", self._safe_source_repr(), exc)
            return None

        if not payload:
            self._logger.debug("Snapshot fetch returned empty payload for %s", self._safe_source_repr())
            return None

        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            self._logger.debug("Snapshot payload decode failed for %s", self._safe_source_repr())
            return None
        return frame

    def _build_snapshot_url(self) -> str:
        assert self._config.stream_url is not None
        if not self._config.snapshot_use_cache_bust:
            return self._config.stream_url

        self._snapshot_counter += 1
        parsed = urllib_parse.urlsplit(self._config.stream_url)
        query = urllib_parse.parse_qsl(parsed.query, keep_blank_values=True)
        query.append(("_ts", str(time.time_ns())))
        query.append(("_seq", str(self._snapshot_counter)))
        return urllib_parse.urlunsplit(parsed._replace(query=urllib_parse.urlencode(query)))
