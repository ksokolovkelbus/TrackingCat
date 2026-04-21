import numpy as np

from app.models import FrameTrackingSummary, OverlayConfig, Track, TrackState
from app.overlay import OverlayRenderer


def test_overlay_renderer_draws_without_errors() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    renderer = OverlayRenderer(OverlayConfig())
    tracks = [
        Track(
            track_id=1,
            display_number=1,
            state=TrackState.CONFIRMED,
            bbox=(40.0, 20.0, 100.0, 80.0),
            square_bbox=(40, 20, 100, 80),
            center_x=70.0,
            center_y=50.0,
            normalized_x=0.4375,
            normalized_y=0.4167,
            confidence=0.88,
            age=3,
            hits=3,
            misses=0,
            consecutive_hits=3,
            consecutive_misses=0,
            first_seen_frame=1,
            last_seen_frame=3,
            last_update_ts=3.0,
        ),
        Track(
            track_id=2,
            display_number=2,
            state=TrackState.CONFIRMED,
            bbox=(10.0, 30.0, 50.0, 70.0),
            square_bbox=(10, 30, 50, 70),
            center_x=30.0,
            center_y=50.0,
            normalized_x=0.1875,
            normalized_y=0.4167,
            confidence=0.75,
            age=3,
            hits=3,
            misses=0,
            consecutive_hits=3,
            consecutive_misses=0,
            first_seen_frame=1,
            last_seen_frame=3,
            last_update_ts=3.0,
        ),
    ]
    summary = FrameTrackingSummary(
        frame_index=3,
        frame_width=160,
        frame_height=120,
        tracking_enabled=True,
        detections_count=2,
        visible_tracks=tracks,
        held_tracks=[],
        tentative_tracks=[],
        lost_tracks=[],
        removed_track_ids=[],
    )

    renderer.draw_square_box(frame, tracks[0].square_bbox)
    renderer.draw_center_marker(frame, tracks[0].center)
    renderer.draw_tracks(frame, tracks)
    renderer.draw_status(frame, "Tracking 2 cat(s)", summary, "connected", "multi-track")
    renderer.draw_fps(frame, 24.8)

    assert frame.shape == (120, 160, 3)
    assert int(frame.sum()) > 0


def test_overlay_can_hide_track_boxes_but_keep_crosshair() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    renderer = OverlayRenderer(
        OverlayConfig(
            show_track_boxes=False,
            show_track_crosshair=True,
        ),
    )
    track = Track(
        track_id=1,
        display_number=1,
        state=TrackState.CONFIRMED,
        bbox=(40.0, 20.0, 100.0, 80.0),
        square_bbox=(40, 20, 100, 80),
        center_x=70.0,
        center_y=50.0,
        normalized_x=0.4375,
        normalized_y=0.4167,
        confidence=0.88,
        age=3,
        hits=3,
        misses=0,
        consecutive_hits=3,
        consecutive_misses=0,
        first_seen_frame=1,
        last_seen_frame=3,
        last_update_ts=3.0,
    )

    renderer.draw_track(frame, track)

    assert int(frame[20, 40].sum()) == 0
    assert int(frame[50, 70].sum()) > 0
