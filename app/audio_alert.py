from __future__ import annotations

import logging
import math
import platform
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

from app.models import SurfaceAlertConfig, SurfaceEvent


@dataclass(frozen=True, slots=True)
class AudioAlertResult:
    played: bool
    backend: str | None = None
    reason: str | None = None


class AudioAlertPlayer:
    def __init__(self, config: SurfaceAlertConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._generated_beep_path: Path | None = None

        self._continuous_stop_event = Event()
        self._continuous_lock = Lock()
        self._continuous_thread: Thread | None = None
        self._continuous_event: SurfaceEvent | None = None

    def _play_once(self, event: SurfaceEvent) -> AudioAlertResult:
        if not self._config.enabled:
            return AudioAlertResult(played=False, reason="disabled")

        first_failure_reason: str | None = None
        if self._config.sound_file:
            sound_path = Path(self._config.sound_file).expanduser()
            file_result = self._play_sound_file(sound_path, event)
            if file_result.played:
                return file_result
            first_failure_reason = file_result.reason
        else:
            first_failure_reason = "sound_file_missing"

        if self._config.beep_fallback:
            beep_result = self._play_beep_fallback(event)
            if beep_result.played:
                return beep_result
            if first_failure_reason is None:
                first_failure_reason = beep_result.reason
            if beep_result.reason is not None and first_failure_reason is not None:
                first_failure_reason = f"{first_failure_reason};{beep_result.reason}"

        reason = first_failure_reason or "no_audio_backend"
        self._logger.warning(
            "surface_alert_audio_failed track_id=%d zone=%s reason=%s",
            event.track_id,
            event.zone_name,
            reason,
        )
        return AudioAlertResult(played=False, reason=reason)

    def play(self, event: SurfaceEvent) -> AudioAlertResult:
        return self._play_once(event)

    def is_playing(self) -> bool:
        with self._continuous_lock:
            return self._continuous_thread is not None and self._continuous_thread.is_alive()

    def start_continuous(self, event: SurfaceEvent) -> None:
        if not self._config.enabled:
            return

        with self._continuous_lock:
            if self._continuous_thread is not None and self._continuous_thread.is_alive():
                return

            self._continuous_stop_event.clear()
            self._continuous_event = event
            self._continuous_thread = Thread(
                target=self._continuous_loop,
                args=(event,),
                daemon=True,
            )
            self._continuous_thread.start()

        self._logger.info(
            "surface_alert_continuous_started track_id=%d zone=%s",
            event.track_id,
            event.zone_name,
        )

    def stop_continuous(self) -> None:
        with self._continuous_lock:
            thread = self._continuous_thread
            if thread is None:
                return
            self._continuous_stop_event.set()

        if thread.is_alive() and thread is not current_thread():
            thread.join(timeout=1.0)

        if platform.system().lower() == "windows":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                self._logger.debug("Failed to stop winsound playback.", exc_info=True)

        with self._continuous_lock:
            self._continuous_thread = None
            self._continuous_event = None
            self._continuous_stop_event.clear()

        self._logger.info("surface_alert_continuous_stopped")

    def close(self) -> None:
        self.stop_continuous()

    def _continuous_loop(self, event: SurfaceEvent) -> None:
        try:
            while not self._continuous_stop_event.is_set():
                self._play_once(event)
                if self._continuous_stop_event.wait(max(0.1, self._config.repeat_interval_seconds)):
                    break
        finally:
            with self._continuous_lock:
                self._continuous_thread = None
                self._continuous_event = None
                self._continuous_stop_event.clear()


    def _play_sound_file(self, sound_path: Path, event: SurfaceEvent) -> AudioAlertResult:
        if not sound_path.exists():
            self._logger.warning(
                "surface_alert_audio_failed track_id=%d zone=%s reason=sound_file_missing path=%s",
                event.track_id,
                event.zone_name,
                sound_path,
            )
            return AudioAlertResult(played=False, reason="sound_file_missing")

        system_name = platform.system().lower()
        if system_name == "windows":
            try:
                import winsound

                winsound.PlaySound(str(sound_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                self._logger.info(
                    "surface_alert_audio_played track_id=%d zone=%s backend=winsound_file",
                    event.track_id,
                    event.zone_name,
                )
                return AudioAlertResult(played=True, backend="winsound_file")
            except Exception:
                self._logger.exception(
                    "surface_alert_audio_failed track_id=%d zone=%s reason=playback_command_failed path=%s",
                    event.track_id,
                    event.zone_name,
                    sound_path,
                )
                return AudioAlertResult(played=False, reason="playback_command_failed")

        for backend_name, command in self._sound_file_commands(sound_path):
            backend_path = shutil.which(command[0])
            if backend_path is None:
                continue
            try:
                subprocess.Popen(
                    [backend_path, *command[1:]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._logger.info(
                    "surface_alert_audio_played track_id=%d zone=%s backend=%s",
                    event.track_id,
                    event.zone_name,
                    backend_name,
                )
                return AudioAlertResult(played=True, backend=backend_name)
            except Exception:
                self._logger.exception(
                    "surface_alert_audio_failed track_id=%d zone=%s reason=playback_command_failed backend=%s",
                    event.track_id,
                    event.zone_name,
                    backend_name,
                )
                return AudioAlertResult(played=False, reason="playback_command_failed")

        self._logger.warning(
            "surface_alert_audio_failed track_id=%d zone=%s reason=backend_unavailable path=%s",
            event.track_id,
            event.zone_name,
            sound_path,
        )
        return AudioAlertResult(played=False, reason="backend_unavailable")

    def _play_beep_fallback(self, event: SurfaceEvent) -> AudioAlertResult:
        system_name = platform.system().lower()
        if system_name == "windows":
            try:
                import winsound

                winsound.MessageBeep()
                self._logger.info(
                    "surface_alert_audio_played track_id=%d zone=%s backend=winsound_beep",
                    event.track_id,
                    event.zone_name,
                )
                return AudioAlertResult(played=True, backend="winsound_beep")
            except Exception:
                self._logger.exception(
                    "surface_alert_audio_failed track_id=%d zone=%s reason=playback_command_failed backend=winsound_beep",
                    event.track_id,
                    event.zone_name,
                )
                return AudioAlertResult(played=False, reason="playback_command_failed")

        if system_name == "darwin":
            mac_sound = Path("/System/Library/Sounds/Ping.aiff")
            if mac_sound.exists():
                return self._spawn_command(
                    event=event,
                    backend_name="afplay_beep",
                    command_name="afplay",
                    args=[str(mac_sound)],
                )

        ffplay_result = self._spawn_command(
            event=event,
            backend_name="ffplay_beep",
            command_name="ffplay",
            args=[
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-f",
                "lavfi",
                "sine=frequency=880:duration=0.18",
            ],
        )
        if ffplay_result.played:
            return ffplay_result

        play_result = self._spawn_command(
            event=event,
            backend_name="play_beep",
            command_name="play",
            args=["-n", "synth", "0.18", "sine", "880"],
        )
        if play_result.played:
            return play_result

        beep_file = self._ensure_generated_beep_file()
        for backend_name, command_name in (("paplay_beep", "paplay"), ("aplay_beep", "aplay")):
            backend_result = self._spawn_command(
                event=event,
                backend_name=backend_name,
                command_name=command_name,
                args=[str(beep_file)],
            )
            if backend_result.played:
                return backend_result

        try:
            print("\a", end="", flush=True)
            self._logger.info(
                "surface_alert_audio_played track_id=%d zone=%s backend=terminal_bell",
                event.track_id,
                event.zone_name,
            )
            return AudioAlertResult(played=True, backend="terminal_bell")
        except Exception:
            self._logger.exception(
                "surface_alert_audio_failed track_id=%d zone=%s reason=backend_unavailable backend=terminal_bell",
                event.track_id,
                event.zone_name,
            )
            return AudioAlertResult(played=False, reason="backend_unavailable")

    def _spawn_command(
        self,
        event: SurfaceEvent,
        backend_name: str,
        command_name: str,
        args: list[str],
    ) -> AudioAlertResult:
        command_path = shutil.which(command_name)
        if command_path is None:
            return AudioAlertResult(played=False, reason="backend_unavailable")
        try:
            subprocess.Popen(
                [command_path, *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._logger.info(
                "surface_alert_audio_played track_id=%d zone=%s backend=%s",
                event.track_id,
                event.zone_name,
                backend_name,
            )
            return AudioAlertResult(played=True, backend=backend_name)
        except Exception:
            self._logger.exception(
                "surface_alert_audio_failed track_id=%d zone=%s reason=playback_command_failed backend=%s",
                event.track_id,
                event.zone_name,
                backend_name,
            )
            return AudioAlertResult(played=False, reason="playback_command_failed")

    def _sound_file_commands(self, sound_path: Path) -> list[tuple[str, list[str]]]:
        system_name = platform.system().lower()
        if system_name == "darwin":
            return [("afplay_file", ["afplay", str(sound_path)])]
        return [
            ("paplay_file", ["paplay", str(sound_path)]),
            ("aplay_file", ["aplay", str(sound_path)]),
            ("ffplay_file", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(sound_path)]),
            ("play_file", ["play", str(sound_path)]),
            ("afplay_file", ["afplay", str(sound_path)]),
        ]

    def _ensure_generated_beep_file(self) -> Path:
        if self._generated_beep_path is not None and self._generated_beep_path.exists():
            return self._generated_beep_path

        with tempfile.NamedTemporaryFile(prefix="trackingcat_beep_", suffix=".wav", delete=False) as temp_file:
            beep_path = Path(temp_file.name)

        sample_rate = 22050
        duration_seconds = 0.18
        amplitude = 12000
        total_frames = int(sample_rate * duration_seconds)
        with wave.open(str(beep_path), "w") as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(sample_rate)
            for frame_index in range(total_frames):
                sample = int(amplitude * math.sin(2.0 * math.pi * 880.0 * (frame_index / sample_rate)))
                wave_file.writeframes(struct.pack("<h", sample))

        self._generated_beep_path = beep_path
        return beep_path
