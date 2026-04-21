from app.models import SceneZone, SceneZonesConfig, Track, TrackState, ZoneType
from app.zones import SceneZoneClassifier, project_zone


def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (20.0, 20.0, 60.0, 80.0),
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


def test_rect_zone_classification_uses_foot_point() -> None:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            zones=[
                SceneZone(
                    name="table",
                    enabled=True,
                    zone_type=ZoneType.SURFACE,
                    shape_type="rect",
                    coordinates_mode="pixels",
                    x1=10,
                    y1=60,
                    x2=80,
                    y2=110,
                ),
            ],
        ),
    )

    match = classifier.classify_track(_make_track(), frame_shape=(120, 120, 3))

    assert match.zone is not None
    assert match.zone.name == "table"


def test_polygon_zone_classification_works() -> None:
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
                    points=((0, 60), (120, 60), (120, 120), (0, 120)),
                ),
            ],
        ),
    )

    match = classifier.classify_track(_make_track(), frame_shape=(120, 120, 3))

    assert match.zone is not None
    assert match.zone.name == "kitchen_floor"
    assert match.zone.zone_type == ZoneType.FLOOR


def test_surface_priority_over_floor_prefers_surface_when_both_match() -> None:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            surface_priority_over_floor=True,
            zones=[
                SceneZone(
                    name="floor",
                    enabled=True,
                    zone_type=ZoneType.FLOOR,
                    shape_type="polygon",
                    coordinates_mode="pixels",
                    points=((0, 60), (120, 60), (120, 120), (0, 120)),
                ),
                SceneZone(
                    name="table",
                    enabled=True,
                    zone_type=ZoneType.SURFACE,
                    shape_type="rect",
                    coordinates_mode="pixels",
                    x1=10,
                    y1=60,
                    x2=80,
                    y2=110,
                ),
            ],
        ),
    )

    match = classifier.classify_track(_make_track(), frame_shape=(120, 120, 3))

    assert match.zone is not None
    assert match.zone.name == "table"
    assert match.zone.zone_type == ZoneType.SURFACE


def test_normalized_zone_scales_to_actual_frame_size() -> None:
    zone = SceneZone(
        name="table",
        enabled=True,
        zone_type=ZoneType.SURFACE,
        shape_type="rect",
        coordinates_mode="normalized",
        x1=0.10,
        y1=0.50,
        x2=0.50,
        y2=1.0,
    )

    small = project_zone(zone, (100, 200, 3))
    large = project_zone(zone, (200, 400, 3))

    assert small.bounds == (20.0, 50.0, 100.0, 100.0)
    assert large.bounds == (40.0, 100.0, 200.0, 200.0)


def test_restricted_zone_is_preferred_over_floor() -> None:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            surface_priority_over_floor=True,
            zones=[
                SceneZone(
                    name="floor",
                    enabled=True,
                    zone_type=ZoneType.FLOOR,
                    shape_type="polygon",
                    coordinates_mode="pixels",
                    points=((0, 60), (120, 60), (120, 120), (0, 120)),
                ),
                SceneZone(
                    name="stove",
                    enabled=True,
                    zone_type=ZoneType.RESTRICTED,
                    shape_type="rect",
                    coordinates_mode="pixels",
                    x1=10,
                    y1=60,
                    x2=80,
                    y2=110,
                ),
            ],
        ),
    )

    match = classifier.classify_track(_make_track(), frame_shape=(120, 120, 3))

    assert match.zone is not None
    assert match.zone.name == "stove"
    assert match.zone.zone_type == ZoneType.RESTRICTED


def test_alert_point_classification_uses_crosshair_center_not_bbox_overlap() -> None:
    classifier = SceneZoneClassifier(
        SceneZonesConfig(
            enabled=True,
            zones=[
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

    match = classifier.classify_alert_track(
        _make_track(bbox=(40.0, 20.0, 100.0, 80.0)),
        frame_shape=(120, 160, 3),
    )

    assert match.zone is None
    assert match.point_inside is False
