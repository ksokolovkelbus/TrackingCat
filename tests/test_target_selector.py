from app.models import Detection
from app.target_selector import select_target


def test_select_target_uses_largest_area_strategy() -> None:
    detections = [
        Detection(15, "cat", 0.60, 10, 10, 30, 30),
        Detection(15, "cat", 0.55, 20, 20, 70, 70),
    ]

    target = select_target(detections, frame_shape=(100, 100, 3), strategy="largest_area")

    assert target is not None
    assert target.detection.area == 2500


def test_select_target_uses_highest_confidence_strategy() -> None:
    detections = [
        Detection(15, "cat", 0.91, 10, 10, 30, 30),
        Detection(15, "cat", 0.72, 20, 20, 70, 70),
    ]

    target = select_target(detections, frame_shape=(100, 100, 3), strategy="highest_confidence")

    assert target is not None
    assert target.detection.confidence == 0.91


def test_select_target_uses_closest_to_center_strategy() -> None:
    detections = [
        Detection(15, "cat", 0.80, 5, 5, 25, 25),
        Detection(15, "cat", 0.70, 40, 40, 60, 60),
        Detection(15, "cat", 0.95, 75, 75, 95, 95),
    ]

    target = select_target(detections, frame_shape=(100, 100, 3), strategy="closest_to_center")

    assert target is not None
    assert target.center == (50, 50)
