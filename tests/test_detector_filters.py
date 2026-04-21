import logging
from types import SimpleNamespace

import numpy as np

from app.detector import YOLODetector
from app.models import Detection, DetectorConfig


def _make_detector(max_frame_area_ratio: float, confidence_threshold: float = 0.2) -> YOLODetector:
    detector = YOLODetector.__new__(YOLODetector)
    detector._config = DetectorConfig(
        model_path='dummy.pt',
        imgsz=640,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.45,
        device='cpu',
        class_name='cat',
        max_frame_area_ratio=max_frame_area_ratio,
    )
    detector._logger = logging.getLogger('test_detector')
    detector._device = 'cpu'
    detector._target_class_name = 'cat'
    detector._target_class_ids = {15}
    detector._normalize_names = YOLODetector._normalize_names
    return detector


def test_oversized_detection_is_filtered_out() -> None:
    detector = _make_detector(max_frame_area_ratio=0.45)
    detection = Detection(class_id=15, class_name='cat', confidence=0.9, x1=0, y1=0, x2=95, y2=95)
    detector._model = SimpleNamespace(
        predict=lambda **kwargs: [SimpleNamespace(names={15: 'cat'}, boxes=[SimpleNamespace()])]
    )
    detector._build_detection = lambda box, names: detection

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result == []


def test_reasonable_detection_is_kept() -> None:
    detector = _make_detector(max_frame_area_ratio=0.45, confidence_threshold=0.5)
    detection = Detection(class_id=15, class_name='cat', confidence=0.9, x1=10, y1=10, x2=40, y2=40)
    detector._model = SimpleNamespace(
        predict=lambda **kwargs: [SimpleNamespace(names={15: 'cat'}, boxes=[SimpleNamespace()])]
    )
    detector._build_detection = lambda box, names: detection

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(result) == 1
    assert result[0].class_name == 'cat'
