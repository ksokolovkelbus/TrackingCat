import logging

from app.models import SceneZone, SceneZonesConfig, SurfaceAlertConfig, Track, TrackState, ZoneType
from app.surface_monitor import SurfaceMonitor
from app.zones import SceneZoneClassifier


def _make_track(*, track_id: int, frame_index: int, last_detection_frame: int, state: TrackState = TrackState.CONFIRMED, center=(50.0, 50.0)) -> Track:
    cx, cy = center
    return Track(
        track_id=track_id,
        display_number=1,
        state=state,
        bbox=(40.0, 40.0, 60.0, 60.0),
        square_bbox=(40, 40, 60, 60),
        center_x=cx,
        center_y=cy,
        normalized_x=0.5,
        normalized_y=0.5,
        confidence=0.9,
        age=1,
        hits=1,
        misses=0,
        consecutive_hits=1,
        consecutive_misses=0,
        first_seen_frame=frame_index,
        last_seen_frame=frame_index,
        last_detection_frame=last_detection_frame,
    )


def test_stale_tracker_only_track_does_not_trigger_surface_alert() -> None:
    zone = SceneZone(name='restricted_1', enabled=True, zone_type=ZoneType.RESTRICTED, shape_type='rect', coordinates_mode='pixels', x1=0, y1=0, x2=100, y2=100)
    classifier = SceneZoneClassifier(SceneZonesConfig(enabled=True, zones=[zone]))
    monitor = SurfaceMonitor(classifier=classifier, config=SurfaceAlertConfig(enabled=True, trigger_from_unknown=True), logger=logging.getLogger('test_surface'))

    stale_track = _make_track(track_id=1, frame_index=10, last_detection_frame=7)
    result = monitor.update([stale_track], frame_shape=(120, 160, 3), frame_index=10, timestamp=1000.0)

    assert result.surface_events == []
    assert result.active_alert_tracks == []


def test_fresh_confirmed_track_can_trigger_surface_alert() -> None:
    zone = SceneZone(name='restricted_1', enabled=True, zone_type=ZoneType.RESTRICTED, shape_type='rect', coordinates_mode='pixels', x1=0, y1=0, x2=100, y2=100)
    classifier = SceneZoneClassifier(SceneZonesConfig(enabled=True, zones=[zone]))
    monitor = SurfaceMonitor(classifier=classifier, config=SurfaceAlertConfig(enabled=True, trigger_from_unknown=True), logger=logging.getLogger('test_surface'))

    fresh_track = _make_track(track_id=1, frame_index=10, last_detection_frame=10)
    result = monitor.update([fresh_track], frame_shape=(120, 160, 3), frame_index=10, timestamp=1000.0)

    assert len(result.surface_events) == 1
    assert result.surface_events[0].zone_name == 'restricted_1'
