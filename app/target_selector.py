from __future__ import annotations

import math
from typing import Iterable

from app.models import Detection, FrameShape, Target, TargetSelectionStrategy
from app.utils import bbox_to_square, normalize_coordinates


class TargetSelector:
    def __init__(self, strategy: TargetSelectionStrategy) -> None:
        self.strategy = strategy

    def select_target(
        self,
        detections: Iterable[Detection],
        frame_shape: FrameShape,
    ) -> Target | None:
        height, width = frame_shape[:2]
        candidates = list(detections)
        if not candidates:
            return None

        selected_detection = self._select_detection(candidates, width, height)
        if selected_detection is None:
            return None

        square_bbox = bbox_to_square(
            selected_detection.x1,
            selected_detection.y1,
            selected_detection.x2,
            selected_detection.y2,
            width,
            height,
        )
        normalized_x, normalized_y = normalize_coordinates(
            selected_detection.center_x,
            selected_detection.center_y,
            width,
            height,
        )
        return Target(
            detection=selected_detection,
            square_bbox=square_bbox,
            center_x=selected_detection.center_x,
            center_y=selected_detection.center_y,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
        )

    def _select_detection(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
    ) -> Detection | None:
        if self.strategy == "largest_area":
            return max(
                detections,
                key=lambda detection: (detection.area, detection.confidence),
            )
        if self.strategy == "highest_confidence":
            return max(
                detections,
                key=lambda detection: (detection.confidence, detection.area),
            )
        if self.strategy == "closest_to_center":
            frame_center_x = frame_width / 2.0
            frame_center_y = frame_height / 2.0
            return min(
                detections,
                key=lambda detection: (
                    math.hypot(
                        detection.center_x - frame_center_x,
                        detection.center_y - frame_center_y,
                    ),
                    -detection.confidence,
                    -detection.area,
                ),
            )
        raise ValueError(f"Unsupported target selection strategy: {self.strategy}")


def select_target(
    detections: Iterable[Detection],
    frame_shape: FrameShape,
    strategy: TargetSelectionStrategy,
) -> Target | None:
    selector = TargetSelector(strategy=strategy)
    return selector.select_target(detections=detections, frame_shape=frame_shape)
