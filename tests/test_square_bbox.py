from app.utils import bbox_to_square


def test_bbox_to_square_builds_expected_square() -> None:
    square_bbox = bbox_to_square(
        x1=10,
        y1=20,
        x2=30,
        y2=50,
        frame_width=100,
        frame_height=100,
    )

    assert square_bbox == (5, 20, 35, 50)


def test_bbox_to_square_stays_inside_frame() -> None:
    square_bbox = bbox_to_square(
        x1=0,
        y1=0,
        x2=20,
        y2=40,
        frame_width=100,
        frame_height=100,
    )

    x1, y1, x2, y2 = square_bbox
    assert square_bbox == (0, 0, 40, 40)
    assert 0 <= x1 < x2 <= 100
    assert 0 <= y1 < y2 <= 100
    assert (x2 - x1) == (y2 - y1)
