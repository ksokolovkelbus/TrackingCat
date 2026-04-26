import logging

from app.models import SceneZone, SceneZonesConfig, SurfaceAlertConfig, Track, TrackState, ZoneType
from app.surface_monitor import SurfaceMonitor
from app.zones import SceneZoneClassifier


class _AudioSpy:
    def __init__(self) -> None:
        self.events: list[tuple[int, str]] = []
        self.continuous_started: list[tuple[int, str]] = []
        self.continuous_stopped = 0

    def play(self, event: object) -> object:
        track_id = getattr(event, "track_id")
        zone_name = getattr(event, "zone_name")
        self.events.append((track_id, zone_name))
        return type("Result", (), {"played": True})()

    def start_continuous(self, event: object) -> None:
        track_id = getattr(event, "track_id")
        zone_name = getattr(event, "zone_name")
        self.continuous_started.append((track_id, zone_name))

    def stop_continuous(self) -> None:
        self.continuous_stopped += 1

    def close(self) -> None:
        return None


def _make_track(
    track_id: int,
    bbox: tuple[float, float, float, float],
) -> Track:
    return Track(
        track_id=track_id,
        display_number=track_id,
        state=TrackState.CONFIRMED,
        bbox=bbox,
        square_bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
        center_x=(bbox[0] + bbox[2]) / 2.0,
        center_y=(bbox[1] + bbox[3]) / 2.0,
        normalized_x=0.0,
        normalized_y=0.0,
        confidence=0.9,
        age=1,
        hits=1,
        misses=0,
        consecutive_hits=1,
        consecutive_misses=0,
        first_seen_frame=1,
        last_seen_frame=1,
        last_update_ts=1.0,
    )


def _make_monitor(**alert_overrides: object) -> tuple[SurfaceMonitor, _AudioSpy]:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            zones=[
                SceneZone(
                    name="kitchen_floor",
                    enabled=True,
                    zone_type=ZoneType.FLOOR,
                    shape_type="polygon",
                    coordinates_mode="pixels",
                    points=((0, 80), (200, 80), (200, 200), (0, 200)),
                ),
                SceneZone(
                    name="table",
                    enabled=True,
                    zone_type=ZoneType.SURFACE,
                    shape_type="rect",
                    coordinates_mode="pixels",
                    x1=20,
                    y1=40,
                    x2=120,
                    y2=110,
                ),
            ],
        ),
    )
    alert_config = SurfaceAlertConfig(**alert_overrides)
    audio_spy = _AudioSpy()
    return (
        SurfaceMonitor(
            classifier=classifier,
            config=alert_config,
            logger=logging.getLogger("test_surface_monitor"),
            audio_player=audio_spy,  # type: ignore[arg-type]
        ),
        audio_spy,
    )


def test_cat_in_floor_zone_does_not_alert() -> None:
    monitor, audio_spy = _make_monitor()

    result = monitor.update(
        tracks=[_make_track(1, (20.0, 90.0, 60.0, 150.0))],
        frame_shape=(200, 200, 3),
        frame_index=1,
        timestamp=1.0,
    )

    assert result.surface_events == []
    assert audio_spy.events == []
    assert result.track_location_states[1].current_zone_type == ZoneType.FLOOR


def test_floor_to_surface_transition_triggers_alert_and_audio() -> None:
    monitor, audio_spy = _make_monitor()
    floor_track = _make_track(1, (20.0, 90.0, 60.0, 150.0))
    surface_track = _make_track(1, (30.0, 30.0, 70.0, 95.0))

    monitor.update([floor_track], frame_shape=(200, 200, 3), frame_index=1, timestamp=1.0)
    result = monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=2, timestamp=2.0)

    assert len(result.surface_events) == 1
    assert result.surface_events[0].zone_name == "table"
    assert audio_spy.events == [(1, "table")]


def test_staying_on_surface_does_not_repeat_alert_every_frame() -> None:
    monitor, audio_spy = _make_monitor()
    floor_track = _make_track(1, (20.0, 90.0, 60.0, 150.0))
    surface_track = _make_track(1, (30.0, 30.0, 70.0, 95.0))

    monitor.update([floor_track], frame_shape=(200, 200, 3), frame_index=1, timestamp=1.0)
    monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=2, timestamp=2.0)
    result = monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=3, timestamp=3.0)

    assert result.surface_events == []
    assert audio_spy.events == [(1, "table")]


def test_surface_to_floor_then_surface_alerts_again() -> None:
    monitor, audio_spy = _make_monitor(cooldown_seconds=0.0, min_interval_per_track=0.0)
    floor_track = _make_track(1, (20.0, 90.0, 60.0, 150.0))
    surface_track = _make_track(1, (30.0, 30.0, 70.0, 95.0))

    monitor.update([floor_track], frame_shape=(200, 200, 3), frame_index=1, timestamp=1.0)
    monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=2, timestamp=2.0)
    monitor.update([floor_track], frame_shape=(200, 200, 3), frame_index=3, timestamp=3.0)
    result = monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=4, timestamp=4.0)

    assert len(result.surface_events) == 1
    assert audio_spy.events == [(1, "table"), (1, "table")]


def test_two_cats_alert_only_for_cat_that_entered_surface() -> None:
    monitor, audio_spy = _make_monitor()
    floor_track_1 = _make_track(1, (20.0, 90.0, 60.0, 150.0))
    floor_track_2 = _make_track(2, (100.0, 90.0, 140.0, 150.0))
    surface_track_2 = _make_track(2, (30.0, 30.0, 70.0, 95.0))

    monitor.update([floor_track_1, floor_track_2], frame_shape=(200, 200, 3), frame_index=1, timestamp=1.0)
    result = monitor.update([floor_track_1, surface_track_2], frame_shape=(200, 200, 3), frame_index=2, timestamp=2.0)

    assert len(result.surface_events) == 1
    assert result.surface_events[0].track_id == 2
    assert audio_spy.events == [(2, "table")]


def test_cooldown_suppresses_repeated_alerts() -> None:
    monitor, audio_spy = _make_monitor(
        cooldown_seconds=10.0,
        min_interval_per_track=10.0,
    )
    floor_track = _make_track(1, (20.0, 90.0, 60.0, 150.0))
    surface_track = _make_track(1, (30.0, 30.0, 70.0, 95.0))

    monitor.update([floor_track], frame_shape=(200, 200, 3), frame_index=1, timestamp=1.0)
    monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=2, timestamp=2.0)
    monitor.update([floor_track], frame_shape=(200, 200, 3), frame_index=3, timestamp=3.0)
    result = monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=4, timestamp=4.0)

    assert result.surface_events == []
    assert len(result.suppressed_surface_events) == 1
    assert audio_spy.events == [(1, "table")]


def test_surface_entry_hysteresis_waits_required_frames() -> None:
    monitor, audio_spy = _make_monitor(
        trigger_from_unknown=True,
        min_zone_frames_before_alert=2,
        cooldown_seconds=0.0,
        min_interval_per_track=0.0,
    )
    surface_track = _make_track(1, (30.0, 30.0, 70.0, 95.0))

    first = monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=1, timestamp=1.0)
    second = monitor.update([surface_track], frame_shape=(200, 200, 3), frame_index=2, timestamp=2.0)

    assert first.surface_events == []
    assert len(second.surface_events) == 1
    assert second.surface_events[0].zone_name == "table"
    assert audio_spy.events == [(1, "table")]


def test_bbox_edge_inside_restricted_but_crosshair_outside_does_not_alert() -> None:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            zones=[
                SceneZone(
                    name="floor",
                    enabled=True,
                    zone_type=ZoneType.FLOOR,
                    shape_type="polygon",
                    coordinates_mode="pixels",
                    points=((0, 0), (200, 0), (200, 200), (0, 200)),
                ),
                SceneZone(
                    name="restricted_1",
                    enabled=True,
                    zone_type=ZoneType.RESTRICTED,
                    shape_type="rect",
                    coordinates_mode="pixels",
                    x1=80,
                    y1=20,
                    x2=120,
                    y2=80,
                ),
            ],
        ),
    )
    audio_spy = _AudioSpy()
    monitor = SurfaceMonitor(
        classifier=classifier,
        config=SurfaceAlertConfig(trigger_only_from_floor=False, trigger_from_unknown=True),
        logger=logging.getLogger("test_surface_monitor"),
        audio_player=audio_spy,  # type: ignore[arg-type]
    )

    result = monitor.update(
        tracks=[_make_track(1, (40.0, 20.0, 100.0, 80.0))],
        frame_shape=(200, 200, 3),
        frame_index=1,
        timestamp=1.0,
    )

    assert result.surface_events == []
    assert audio_spy.events == []
    assert result.track_location_states[1].current_zone_type == ZoneType.FLOOR


def test_crosshair_inside_restricted_zone_triggers_alert() -> None:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            zones=[
                SceneZone(
                    name="floor",
                    enabled=True,
                    zone_type=ZoneType.FLOOR,
                    shape_type="polygon",
                    coordinates_mode="pixels",
                    points=((0, 0), (200, 0), (200, 200), (0, 200)),
                ),
                SceneZone(
                    name="restricted_1",
                    enabled=True,
                    zone_type=ZoneType.RESTRICTED,
                    shape_type="rect",
                    coordinates_mode="pixels",
                    x1=80,
                    y1=20,
                    x2=140,
                    y2=90,
                ),
            ],
        ),
    )
    audio_spy = _AudioSpy()
    monitor = SurfaceMonitor(
        classifier=classifier,
        config=SurfaceAlertConfig(trigger_only_from_floor=False, trigger_from_unknown=True),
        logger=logging.getLogger("test_surface_monitor"),
        audio_player=audio_spy,  # type: ignore[arg-type]
    )

    result = monitor.update(
        tracks=[_make_track(1, (80.0, 20.0, 140.0, 80.0))],
        frame_shape=(200, 200, 3),
        frame_index=1,
        timestamp=1.0,
    )

    assert len(result.surface_events) == 1
    assert result.surface_events[0].zone_name == "restricted_1"
    assert result.track_location_states[1].current_zone_type == ZoneType.RESTRICTED
    assert audio_spy.events == [(1, "restricted_1")]
