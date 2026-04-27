from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import cv2
import numpy as np

from app.models import PanTiltControlConfig


@dataclass(slots=True)
class PanTiltState:
    pan_angle: int = 90
    tilt_angle: int = 90
    laser_on: bool = False
    step_degrees: int = 3
    speed_mode: str = "medium"
    connected: bool = False
    wifi_mode: str | None = None
    ip_address: str | None = None
    last_command: str = "idle"
    last_error: str | None = None
    last_updated_ts: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlButton:
    action: str
    label: str
    rect: tuple[int, int, int, int]
    accent: bool = False
    toggled: bool = False

    def contains(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom


class PanTiltController:
    def __init__(self, config: PanTiltControlConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._state = PanTiltState(step_degrees=config.default_step_degrees)
        self._last_poll_monotonic = 0.0

    @property
    def state(self) -> PanTiltState:
        return self._state

    def _resolve_direction(self, direction: str) -> str:
        button_to_direction = {
            "up": self._config.button_up_direction,
            "down": self._config.button_down_direction,
            "left": self._config.button_left_direction,
            "right": self._config.button_right_direction,
        }
        return button_to_direction[direction]


    def maybe_refresh_state(self, force: bool = False) -> PanTiltState:
        now = time.monotonic()
        if not force and (now - self._last_poll_monotonic) < self._config.status_poll_interval_seconds:
            return self._state
        self._last_poll_monotonic = now
        try:
            payload = self._request_json("/api/state")
        except RuntimeError as exc:
            self._state.connected = False
            self._state.last_error = str(exc)
            return self._state
        self._merge_state(payload, fallback_command="state")
        return self._state

    def move(self, direction: str) -> PanTiltState:
        resolved = self._resolve_direction(direction)
        pan_delta = 0
        tilt_delta = 0
        step = self._state.step_degrees
        if resolved == "left":
            pan_delta = -step
        elif resolved == "right":
            pan_delta = step
        elif resolved == "up":
            tilt_delta = -step
        elif resolved == "down":
            tilt_delta = step
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        payload = self._request_json("/api/move", {"pan_delta": pan_delta, "tilt_delta": tilt_delta})
        self._merge_state(payload, fallback_command=f"move:{direction}")
        return self._state

    def center(self) -> PanTiltState:
        payload = self._request_json("/api/center")
        self._merge_state(payload, fallback_command="center")
        return self._state

    def stop(self) -> PanTiltState:
        payload = self._request_json("/api/stop")
        self._merge_state(payload, fallback_command="stop")
        return self._state

    def start_continuous_move(self, direction: str) -> PanTiltState:
        resolved = self._resolve_direction(direction)
        payload = self._request_json("/api/start_move", {"direction": resolved})
        self._merge_state(payload, fallback_command=f"start:{direction}")
        return self._state

    def toggle_laser(self) -> PanTiltState:
        payload = self._request_json("/api/laser/toggle")
        self._merge_state(payload, fallback_command="laser-toggle")
        return self._state

    def set_step(self, step_degrees: int) -> PanTiltState:
        payload = self._request_json("/api/step", {"degrees": int(step_degrees)})
        self._merge_state(payload, fallback_command=f"step:{step_degrees}")
        return self._state

    def set_speed_mode(self, mode: str) -> PanTiltState:
        payload = self._request_json("/api/speed", {"mode": mode})
        self._merge_state(payload, fallback_command=f"speed:{mode}")
        return self._state

    def execute_action(self, action: str) -> PanTiltState:
        if action in {"left", "right", "up", "down"}:
            return self.move(action)
        if action == "center":
            return self.center()
        if action == "laser-toggle":
            return self.toggle_laser()
        if action == "stop":
            return self.stop()
        if action.startswith("speed:"):
            return self.set_speed_mode(action.split(":", 1)[1])
        if action.startswith("step:"):
            return self.set_step(int(action.split(":", 1)[1]))
        if action == "refresh":
            return self.maybe_refresh_state(force=True)
        raise ValueError(f"Unsupported action: {action}")

    def _request_json(self, path: str, params: dict[str, int] | None = None) -> dict[str, object]:
        url = self._build_url(path, params)
        request = urllib_request.Request(url, headers={"Cache-Control": "no-cache"})
        try:
            with urllib_request.urlopen(request, timeout=self._config.request_timeout_seconds) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except (urllib_error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"PanTilt request failed: {exc}") from exc
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PanTilt controller returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("PanTilt controller returned unexpected payload.")
        return data

    def _build_url(self, path: str, params: dict[str, int] | None = None) -> str:
        base = self._config.base_url.rstrip("/")
        query = urllib_parse.urlencode(params or {})
        return f"{base}{path}{('?' + query) if query else ''}"

    def _merge_state(self, payload: dict[str, object], fallback_command: str) -> None:
        self._state.pan_angle = int(payload.get("pan_angle", self._state.pan_angle))
        self._state.tilt_angle = int(payload.get("tilt_angle", self._state.tilt_angle))
        self._state.laser_on = bool(payload.get("laser_on", self._state.laser_on))
        self._state.step_degrees = int(payload.get("step_degrees", self._state.step_degrees))
        self._state.speed_mode = str(payload.get("speed_mode", self._state.speed_mode))
        self._state.connected = bool(payload.get("connected", True))
        self._state.wifi_mode = str(payload.get("wifi_mode")) if payload.get("wifi_mode") is not None else None
        self._state.ip_address = str(payload.get("ip_address")) if payload.get("ip_address") is not None else None
        self._state.last_command = str(payload.get("last_command", fallback_command))
        self._state.last_error = str(payload.get("error")) if payload.get("error") else None
        self._state.last_updated_ts = time.time()


class PanTiltControlOverlay:
    def __init__(self, config: PanTiltControlConfig) -> None:
        self._config = config
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def build_buttons(self, frame_shape: tuple[int, int] | tuple[int, int, int], state: PanTiltState) -> list[ControlButton]:
        height, width = frame_shape[:2]
        size = 72
        gap = 12
        base_x = width - (size * 3 + gap * 4)
        base_y = height - (size * 3 + gap * 4)
        step_buttons = [
            ControlButton("speed:slow", "SLOW", (base_x, base_y - size - gap, base_x + size, base_y), toggled=state.speed_mode == "slow"),
            ControlButton(
                "speed:medium",
                "MED",
                (base_x + size + gap, base_y - size - gap, base_x + size * 2 + gap, base_y),
                toggled=state.speed_mode == "medium",
            ),
            ControlButton(
                "speed:fast",
                "FAST",
                (base_x + (size + gap) * 2, base_y - size - gap, base_x + size * 3 + gap * 2, base_y),
                toggled=state.speed_mode == "fast",
            ),
        ]
        dpad = [
            ControlButton("up", "UP", (base_x + size + gap, base_y, base_x + size * 2 + gap, base_y + size), accent=True),
            ControlButton("left", "LEFT", (base_x, base_y + size + gap, base_x + size, base_y + size * 2 + gap), accent=True),
            ControlButton("center", "C", (base_x + size + gap, base_y + size + gap, base_x + size * 2 + gap, base_y + size * 2 + gap), toggled=True),
            ControlButton("right", "RIGHT", (base_x + (size + gap) * 2, base_y + size + gap, base_x + size * 3 + gap * 2, base_y + size * 2 + gap), accent=True),
            ControlButton("down", "DOWN", (base_x + size + gap, base_y + (size + gap) * 2, base_x + size * 2 + gap, base_y + size * 3 + gap * 2), accent=True),
        ]
        aux = [
            ControlButton("laser-toggle", "LASER", (24, height - 96, 180, height - 36), accent=state.laser_on, toggled=state.laser_on),
            ControlButton("refresh", "REFRESH", (196, height - 96, 352, height - 36)),
            ControlButton("stop", "STOP", (368, height - 96, 524, height - 36), accent=True),
        ]
        return step_buttons + dpad + aux

    def draw(self, frame: np.ndarray, state: PanTiltState, fps: float, source_status: str) -> list[ControlButton]:
        buttons = self.build_buttons(frame.shape, state)
        self._draw_header(frame, state=state, fps=fps, source_status=source_status)
        for button in buttons:
            self._draw_button(frame, button)
        return buttons

    def find_button(self, buttons: list[ControlButton], x: int, y: int) -> ControlButton | None:
        for button in buttons:
            if button.contains(x, y):
                return button
        return None

    def _draw_header(self, frame: np.ndarray, state: PanTiltState, fps: float, source_status: str) -> None:
        lines = [
            "iPad PanTilt manual mode",
            f"source: {source_status} | fps: {fps:.1f}",
            f"pan: {state.pan_angle} deg | tilt: {state.tilt_angle} deg | speed: {state.speed_mode.upper()} | laser: {'ON' if state.laser_on else 'OFF'}",
            f"controller: {'connected' if state.connected else 'offline'} | net: {state.wifi_mode or '-'} | ip: {state.ip_address or '-'}",
            "hold mouse on arrows = continuous move | keys: WASD 1/3/8 L C R Q",
        ]
        x = 16
        y = 24
        for index, text in enumerate(lines):
            text_size, baseline = cv2.getTextSize(text, self._font, 0.62, 2 if index == 0 else 1)
            left = x - 8
            top = y - text_size[1] - 8
            right = x + text_size[0] + 8
            bottom = y + baseline + 4
            cv2.rectangle(frame, (left, top), (right, bottom), (16, 16, 16), thickness=-1)
            color = (80, 220, 120) if index == 0 else (240, 240, 240)
            if index == 3 and not state.connected:
                color = (0, 80, 255)
            cv2.putText(frame, text, (x, y), self._font, 0.62, color, 2 if index == 0 else 1, cv2.LINE_AA)
            y += 24
        if state.last_error:
            self._draw_footer_notice(frame, f"last error: {state.last_error}", color=(0, 80, 255))

    def _draw_footer_notice(self, frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        text_size, baseline = cv2.getTextSize(text, self._font, 0.58, 1)
        x = 16
        y = frame.shape[0] - 110
        cv2.rectangle(frame, (x - 8, y - text_size[1] - 8), (x + text_size[0] + 8, y + baseline + 4), (16, 16, 16), thickness=-1)
        cv2.putText(frame, text, (x, y), self._font, 0.58, color, 1, cv2.LINE_AA)

    def _draw_button(self, frame: np.ndarray, button: ControlButton) -> None:
        left, top, right, bottom = button.rect
        fill = (30, 30, 30)
        border = (120, 120, 120)
        text_color = (240, 240, 240)
        if button.accent:
            border = (0, 200, 255)
        if button.toggled:
            fill = (38, 78, 38) if button.action.startswith("step:") or button.action == "center" else (0, 0, 120)
            border = (80, 220, 120) if button.action.startswith("step:") else (0, 180, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), fill, thickness=-1)
        cv2.rectangle(frame, (left, top), (right, bottom), border, thickness=2)
        text_size, baseline = cv2.getTextSize(button.label, self._font, 0.9, 2)
        text_x = left + ((right - left) - text_size[0]) // 2
        text_y = top + ((bottom - top) + text_size[1]) // 2
        cv2.putText(frame, button.label, (text_x, text_y), self._font, 0.9, text_color, 2, cv2.LINE_AA)
