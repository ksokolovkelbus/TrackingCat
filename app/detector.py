from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.models import Detection, DetectorConfig


class DetectorError(RuntimeError):
    """Raised when the YOLO detector cannot be initialized."""


class YOLODetector:
    def __init__(self, config: DetectorConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._device = self._resolve_device(config.device)
        self._yolo = self._import_yolo()

        try:
            self._model = self._yolo(config.model_path)
        except Exception as exc:
            raise DetectorError(
                f"Failed to load YOLO model from '{config.model_path}'."
            ) from exc

        self._target_class_name = config.class_name.strip().lower()
        model_names = self._normalize_names(getattr(self._model, "names", {}))
        self._target_class_ids = {
            class_id
            for class_id, class_name in model_names.items()
            if class_name.strip().lower() == self._target_class_name
        }
        if not self._target_class_ids:
            self._logger.warning(
                "Class '%s' is not present in the loaded model labels. "
                "Detections for that class will stay empty.",
                config.class_name,
            )
        self._logger.info(
            "Detector initialized: model=%s, device=%s, imgsz=%d, conf=%.3f, iou=%.3f, class=%s",
            config.model_path,
            self._device,
            config.imgsz,
            config.confidence_threshold,
            config.iou_threshold,
            config.class_name,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            self._logger.warning("Detector received an empty frame.")
            return []

        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_width * frame_height)

        try:
            results = self._model.predict(
                source=frame,
                imgsz=self._config.imgsz,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception:
            self._logger.exception("YOLO inference failed.")
            return []

        if not results:
            return []

        result = results[0]
        names = self._normalize_names(getattr(result, "names", {}) or getattr(self._model, "names", {}))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[Detection] = []
        for box in boxes:
            detection = self._build_detection(box=box, names=names)
            if detection is None:
                continue
            if self._target_class_ids and detection.class_id not in self._target_class_ids:
                continue
            if detection.class_name.strip().lower() != self._target_class_name:
                continue
            if detection.confidence < self._config.confidence_threshold:
                continue
            if frame_area > 0:
                detection_area_ratio = detection.area / frame_area
                if detection_area_ratio > self._config.max_frame_area_ratio:
                    self._logger.debug(
                        "Skipping oversized detection: class=%s conf=%.3f area_ratio=%.3f max_allowed=%.3f bbox=(%.1f, %.1f, %.1f, %.1f)",
                        detection.class_name,
                        detection.confidence,
                        detection_area_ratio,
                        self._config.max_frame_area_ratio,
                        detection.x1,
                        detection.y1,
                        detection.x2,
                        detection.y2,
                    )
                    continue
            detections.append(detection)
        return detections

    def _build_detection(self, box: Any, names: dict[int, str]) -> Detection | None:
        try:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
        except Exception:
            self._logger.debug("Skipping malformed detection box.", exc_info=True)
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        class_name = names.get(class_id, str(class_id))
        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
        )

    def _resolve_device(self, requested_device: str) -> str:
        if not requested_device.lower().startswith("cuda"):
            return requested_device

        try:
            import torch
        except ImportError:
            self._logger.warning("CUDA device requested, but torch is not available. Falling back to CPU.")
            return "cpu"

        if not torch.cuda.is_available():
            self._logger.warning("CUDA device requested, but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return requested_device

    @staticmethod
    def _import_yolo() -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "ultralytics is not installed. Install dependencies from requirements.txt."
            ) from exc
        return YOLO

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        return {}
