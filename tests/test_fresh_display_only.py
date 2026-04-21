from app.models import Track, TrackState


def _track(last_detection_frame: int, last_seen_frame: int, state: TrackState = TrackState.CONFIRMED) -> Track:
    return Track(
        track_id=1,
        display_number=1,
        state=state,
        bbox=(10.0, 10.0, 20.0, 20.0),
        square_bbox=(10, 10, 20, 20),
        center_x=15.0,
        center_y=15.0,
        normalized_x=0.5,
        normalized_y=0.5,
        confidence=0.9,
        age=1,
        hits=1,
        misses=0,
        consecutive_hits=1,
        consecutive_misses=0,
        first_seen_frame=1,
        last_seen_frame=last_seen_frame,
        last_detection_frame=last_detection_frame,
    )


def test_confirmed_track_is_displayed_only_on_fresh_detection_frame() -> None:
    fresh = _track(last_detection_frame=10, last_seen_frame=10)
    stale = _track(last_detection_frame=8, last_seen_frame=10)

    assert fresh.is_displayed(10) is True
    assert stale.is_displayed(10) is False


def test_held_track_is_not_displayed_as_target() -> None:
    held = _track(last_detection_frame=10, last_seen_frame=10, state=TrackState.HELD)
    assert held.is_displayed(10) is False
