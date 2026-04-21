from __future__ import annotations

import json
import logging
import ssl
import subprocess
from pathlib import Path
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from string import Template
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.models import SourceConfig


class BrowserCameraIngestServer:
    def __init__(self, config: "SourceConfig", logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_jpeg: bytes | None = None
        self._latest_frame_id = 0
        self._latest_frame_ts = 0.0

    @property
    def public_url(self) -> str:
        return f"https://{self._config.browser_public_host}:{self._config.browser_camera_port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        cert_path, key_path = self._ensure_tls_certificate()
        server = ThreadingHTTPServer(
            (self._config.browser_camera_host, self._config.browser_camera_port),
            self._build_handler(),
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        server.socket = context.wrap_socket(server.socket, server_side=True)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="browser-camera-ingest", daemon=True)
        self._thread.start()
        self._logger.info(
            "Browser camera ingest started on %s:%d (open %s)",
            self._config.browser_camera_host,
            self._config.browser_camera_port,
            self.public_url,
        )

    def _ensure_tls_certificate(self) -> tuple[Path, Path]:
        project_root = Path(__file__).resolve().parent.parent
        cert_dir = project_root / "certs"
        cert_dir.mkdir(parents=True, exist_ok=True)
        cert_path = cert_dir / "browser_camera.crt"
        key_path = cert_dir / "browser_camera.key"
        if cert_path.exists() and key_path.exists():
            return cert_path, key_path

        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-days",
                "365",
                "-subj",
                f"/CN={self._config.browser_public_host}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return cert_path, key_path

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        try:
            server.shutdown()
            server.server_close()
        finally:
            thread = self._thread
            self._thread = None
            if thread is not None:
                thread.join(timeout=2.0)

    def submit_jpeg(self, payload: bytes) -> bool:
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return False
        with self._lock:
            self._latest_frame = frame
            self._latest_frame_jpeg = payload
            self._latest_frame_id += 1
            self._latest_frame_ts = time.time()
        return True

    def get_latest_frame(self) -> tuple[int, np.ndarray | None, float]:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            return self._latest_frame_id, frame, self._latest_frame_ts

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_frame_jpeg

    def _build_handler(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                server_ref._logger.debug("browser-camera http: " + fmt, *args)

            def do_GET(self) -> None:
                if self.path == "/" or self.path.startswith("/?"):
                    self._send_html(server_ref._build_index_html())
                    return
                if self.path == "/health":
                    frame_id, frame, ts = server_ref.get_latest_frame()
                    age = None if ts <= 0 else max(0.0, time.time() - ts)
                    self._send_json(
                        {
                            "ok": True,
                            "frame_id": frame_id,
                            "has_frame": frame is not None,
                            "frame_age_seconds": age,
                            "public_url": server_ref.public_url,
                        }
                    )
                    return
                if self.path == "/snapshot.jpg":
                    jpeg = server_ref.get_latest_jpeg()
                    if jpeg is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No browser frame uploaded yet")
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:
                if self.path != "/api/frame":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                    return
                if content_length <= 0:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Empty body")
                    return
                payload = self.rfile.read(content_length)
                if not server_ref.submit_jpeg(payload):
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JPEG payload")
                    return
                self._send_json({"ok": True, "frame_id": server_ref._latest_frame_id})

            def _send_html(self, body: str) -> None:
                payload = body.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(self, data: dict[str, object]) -> None:
                payload = json.dumps(data).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    def _build_index_html(self) -> str:
        width = max(160, int(self._config.browser_capture_width))
        height = max(120, int(self._config.browser_capture_height))
        interval_ms = max(80, int(self._config.browser_capture_interval_ms))
        jpeg_quality = min(0.95, max(0.2, float(self._config.browser_jpeg_quality) / 100.0))
        template = Template(
            """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TrackingCat Browser Camera</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
    .wrap { max-width: 840px; margin: 0 auto; }
    video { width: 100%; max-width: 720px; background: #000; border-radius: 12px; }
    button, select { font-size: 16px; padding: 10px 12px; margin: 6px 6px 6px 0; }
    .row { margin: 12px 0; }
    .muted { color: #bbb; }
    code { background: #222; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>TrackingCat Browser Camera</h1>
  <p class="muted">Open this page from a phone, tablet, or laptop in the same local network. Then press Start.</p>
  <div class="row">
    <label for="cameraSelect">Camera:</label>
    <select id="cameraSelect"></select>
  </div>
  <div class="row">
    <button id="startBtn">Start camera</button>
    <button id="stopBtn" disabled>Stop</button>
  </div>
  <div class="row">
    <video id="preview" autoplay playsinline muted></video>
    <canvas id="buffer" width="$width" height="$height" hidden></canvas>
  </div>
  <div class="row muted">Upload interval: <code>$interval_ms ms</code>, JPEG quality: <code>$jpeg_quality_display</code>, target size: <code>$width x $height</code></div>
  <div class="row"><strong id="status">Idle</strong></div>
</div>
<script>
const TARGET_WIDTH = $width;
const TARGET_HEIGHT = $height;
const UPLOAD_INTERVAL_MS = $interval_ms;
const JPEG_QUALITY = $jpeg_quality;

const preview = document.getElementById("preview");
const canvas = document.getElementById("buffer");
const ctx = canvas.getContext("2d", { alpha: false });
const cameraSelect = document.getElementById("cameraSelect");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
let stream = null;
let timer = null;
let busy = false;

async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(device => device.kind === "videoinput");
  cameraSelect.innerHTML = "";
  cameras.forEach((device, index) => {
    const opt = document.createElement("option");
    opt.value = device.deviceId;
    opt.textContent = device.label || `Camera $${index + 1}`;
    cameraSelect.appendChild(opt);
  });
}

async function startCamera() {
  stopCamera();
  const constraints = {
    audio: false,
    video: {
      deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
      facingMode: "environment",
      width: { ideal: TARGET_WIDTH },
      height: { ideal: TARGET_HEIGHT }
    }
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  preview.srcObject = stream;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "Camera active, uploading frames...";
  timer = setInterval(captureAndSend, UPLOAD_INTERVAL_MS);
  await listCameras();
}

function stopCamera() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  preview.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Stopped";
}

async function captureAndSend() {
  if (!stream || busy) {
    return;
  }
  if (!preview.videoWidth || !preview.videoHeight) {
    return;
  }
  busy = true;
  try {
    ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
    if (!blob) {
      throw new Error("toBlob failed");
    }
    const resp = await fetch("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
      cache: "no-store"
    });
    if (!resp.ok) {
      throw new Error(`upload failed: $${resp.status}`);
    }
    statusEl.textContent = `Uploading, preview $${preview.videoWidth}x$${preview.videoHeight}, sent $${canvas.width}x$${canvas.height}`;
  } catch (err) {
    statusEl.textContent = `Upload error: $${err}`;
  } finally {
    busy = false;
  }
}

startBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (err) {
    statusEl.textContent = `Camera start failed: $${err}`;
  }
});
stopBtn.addEventListener("click", stopCamera);

(async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusEl.textContent = "This browser does not support camera capture on this page.";
    return;
  }
  try {
    await listCameras();
  } catch (err) {
    statusEl.textContent = `Cannot enumerate cameras: $${err}`;
  }
})();
</script>
</body>
</html>
"""
        )
        return template.substitute(
            width=width,
            height=height,
            interval_ms=interval_ms,
            jpeg_quality=jpeg_quality,
            jpeg_quality_display=self._config.browser_jpeg_quality,
        )
