import logging
from unittest.mock import patch

from app.audio_alert import AudioAlertPlayer
from app.models import SurfaceAlertConfig, SurfaceEvent, ZoneType


def test_audio_alert_uses_ffplay_beep_fallback_when_no_sound_file() -> None:
    player = AudioAlertPlayer(
        config=SurfaceAlertConfig(enabled=True, sound_file=None, beep_fallback=True),
        logger=logging.getLogger("test_audio_alert"),
    )
    event = SurfaceEvent(
        event_type="entered_surface",
        track_id=7,
        zone_name="table",
        zone_type=ZoneType.SURFACE,
        frame_index=10,
        timestamp=10.0,
        message="ALERT: Cat on table",
    )

    with (
        patch("app.audio_alert.shutil.which") as which_mock,
        patch("app.audio_alert.subprocess.Popen") as popen_mock,
    ):
        which_mock.side_effect = lambda name: f"/usr/bin/{name}" if name == "ffplay" else None
        result = player.play(event)

    assert result.played is True
    assert result.backend == "ffplay_beep"
    popen_mock.assert_called_once()
