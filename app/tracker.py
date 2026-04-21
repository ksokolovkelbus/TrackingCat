from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace

import cv2
import numpy as np

from app.models import (
    Detection,
    FloatBBox,
    FrameShape,
    FrameTrackingSummary,
    Track,
    TrackingConfig,
    TrackingPipelineState,
    TrackState,
)
from app.utils import (
    area_ratio,
    bbox_area,
    bbox_iou,
    bbox_to_square,
    calculate_center_from_bbox,
    center_distance,
    clamp_float_bbox,
    normalize_coordinates,
    smooth_bbox,
    smooth_value,
    sort_tracks_for_display,
)


class FrameTrackerWrapper:
    def initialize(self, frame: np.ndarray, bbox: FloatBBox) -> bool:
        raise NotImplementedError

    def update(self, frame: np.ndarray) -> tuple[bool, FloatBBox]:
        raise NotImplementedError


class OpenCvFrameTracker(FrameTrackerWrapper):
    _BACKEND_PRIORITY = ("CSRT", "KCF", "MIL")

    def __init__(self) -> None:
        tracker_bundle = self._create_tracker()
        self._tracker = tracker_bundle[0] if tracker_bundle is not None else None
        self.backend_name = tracker_bundle[1] if tracker_bundle is not None else None

    @classmethod
    def _create_tracker(cls) -> tuple[object, str] | None:
        for backend_name in cls._BACKEND_PRIORITY:
            factory = cls._resolve_backend_factory(backend_name)
            if factory is None:
                continue
            try:
                return factory(), backend_name
            except Exception:
                continue
        return None

    @staticmethod
    def _resolve_backend_factory(backend_name: str) -> Callable[[], object] | None:
        factory_name = f"Tracker{backend_name}_create"
        direct_factory = getattr(cv2, factory_name, None)
        if callable(direct_factory):
            return direct_factory

        legacy_module = getattr(cv2, "legacy", None)
        legacy_factory = getattr(legacy_module, factory_name, None) if legacy_module is not None else None
        if callable(legacy_factory):
            return legacy_factory
        return None

    @staticmethod
    def _sanitize_bbox(frame: np.ndarray, bbox: FloatBBox) -> tuple[int, int, int, int] | None:
        if frame is None or frame.size == 0:
            return None

        frame_height, frame_width = frame.shape[:2]
        x1_raw, y1_raw, x2_raw, y2_raw = bbox

        x1 = int(round(min(x1_raw, x2_raw)))
        y1 = int(round(min(y1_raw, y2_raw)))
        x2 = int(round(max(x1_raw, x2_raw)))
        y2 = int(round(max(y1_raw, y2_raw)))

        x1 = max(0, min(x1, frame_width - 2))
        y1 = max(0, min(y1, frame_height - 2))
        x2 = max(x1 + 1, min(x2, frame_width - 1))
        y2 = max(y1 + 1, min(y2, frame_height - 1))

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None
        return (x1, y1, width, height)

    @classmethod
    def _sanitize_updated_bbox(
        cls,
        frame: np.ndarray,
        tracker_bbox: tuple[float, float, float, float],
    ) -> FloatBBox | None:
        x, y, width, height = tracker_bbox
        sanitized = cls._sanitize_bbox(
            frame,
            (float(x), float(y), float(x + width), float(y + height)),
        )
        if sanitized is None:
            return None
        left, top, box_width, box_height = sanitized
        return (
            float(left),
            float(top),
            float(left + box_width),
            float(top + box_height),
        )

    def initialize(self, frame: np.ndarray, bbox: FloatBBox) -> bool:
        if self._tracker is None:
            return False

        tracker_bbox = self._sanitize_bbox(frame, bbox)
        if tracker_bbox is None:
            return False

        try:
            result = self._tracker.init(frame, tracker_bbox)
        except Exception:
            return False
        return True if result is None else bool(result)

    def update(self, frame: np.ndarray) -> tuple[bool, FloatBBox]:
        if self._tracker is None:
            return False, (0.0, 0.0, 0.0, 0.0)

        try:
            ok, tracker_bbox = self._tracker.update(frame)
        except Exception:
            return False, (0.0, 0.0, 0.0, 0.0)

        if not ok:
            return False, (0.0, 0.0, 0.0, 0.0)

        sanitized_bbox = self._sanitize_updated_bbox(frame, tracker_bbox)
        if sanitized_bbox is None:
            return False, (0.0, 0.0, 0.0, 0.0)
        return True, sanitized_bbox


OpenCvMILFrameTracker = OpenCvFrameTracker


class MultiCatTracker:
    _VELOCITY_ALPHA = 0.5

    def __init__(self, config: TrackingConfig, logger: logging.Logger) -> None:
        if not config.tracking_enabled:
            raise ValueError("MultiCatTracker requires tracking_enabled=true.")
        self._config = config
        self._logger = logger
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: list[Detection],
        frame_shape: FrameShape,
        frame_index: int,
        timestamp: float | None = None,
    ) -> FrameTrackingSummary:
        return self._run_detection_update(
            detections=detections,
            frame_shape=frame_shape,
            frame_index=frame_index,
            timestamp=timestamp,
        )

    def refresh_with_detections(
        self,
        detections: list[Detection],
        frame_shape: FrameShape,
        frame_index: int,
        timestamp: float | None = None,
    ) -> FrameTrackingSummary:
        return self.update(
            detections=detections,
            frame_shape=frame_shape,
            frame_index=frame_index,
            timestamp=timestamp,
        )

    def update_from_tracker(
        self,
        track_id: int,
        bbox: FloatBBox,
        frame_shape: FrameShape,
        frame_index: int,
        timestamp: float | None = None,
    ) -> bool:
        track = self._tracks.get(track_id)
        if track is None or track.state == TrackState.REMOVED:
            return False

        frame_height, frame_width = frame_shape[:2]
        now = timestamp if timestamp is not None else time.time()
        previous_state = track.state
        previous_center_x = track.center_x
        previous_center_y = track.center_y
        tracker_bbox = clamp_float_bbox(bbox, frame_width, frame_height)
        smoothed_bbox = smooth_bbox(track.bbox, tracker_bbox, self._config.smoothing_alpha)
        smoothed_bbox = clamp_float_bbox(smoothed_bbox, frame_width, frame_height)

        self._update_track_geometry(
            track=track,
            bbox=smoothed_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        track.velocity_x = smooth_value(
            track.velocity_x,
            track.center_x - previous_center_x,
            self._VELOCITY_ALPHA,
        )
        track.velocity_y = smooth_value(
            track.velocity_y,
            track.center_y - previous_center_y,
            self._VELOCITY_ALPHA,
        )
        track.age += 1
        track.hits += 1
        track.consecutive_hits += 1
        track.consecutive_misses = 0
        track.last_seen_frame = frame_index
        track.last_update_ts = now
        track.predicted_bbox = None
        track.held_frames = 0
        track.reconfirm_hits = 0
        track.tracker_only_updates += 1
        track.tracker_failures = 0

        if previous_state in {TrackState.HELD, TrackState.LOST}:
            track.state = TrackState.CONFIRMED
            self._logger.info("track_reacquired track_id=%d", track.track_id)
        elif previous_state == TrackState.TENTATIVE:
            track.state = TrackState.TENTATIVE
        else:
            track.state = TrackState.CONFIRMED
        return True

    def mark_tracker_failure(
        self,
        track_id: int,
        frame_shape: FrameShape,
        timestamp: float | None = None,
    ) -> bool:
        track = self._tracks.get(track_id)
        if track is None or track.state == TrackState.REMOVED:
            return False

        track.tracker_failures += 1
        frame_height, frame_width = frame_shape[:2]
        now = timestamp if timestamp is not None else time.time()
        return self._handle_unmatched_track(
            track=track,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=now,
        )

    def build_summary(
        self,
        frame_shape: FrameShape,
        frame_index: int,
        detections_count: int,
        removed_track_ids: list[int] | None = None,
    ) -> FrameTrackingSummary:
        frame_height, frame_width = frame_shape[:2]
        visible_tracks = self._assign_display_numbers(frame_index)
        held_tracks = [track for track in visible_tracks if track.state == TrackState.HELD]
        tentative_tracks = sorted(
            (
                track
                for track in self._iter_non_removed_tracks()
                if track.state == TrackState.TENTATIVE
            ),
            key=lambda track: track.track_id,
        )
        lost_tracks = sorted(
            (
                track
                for track in self._iter_non_removed_tracks()
                if track.state == TrackState.LOST
            ),
            key=lambda track: track.track_id,
        )
        return FrameTrackingSummary(
            frame_index=frame_index,
            frame_width=frame_width,
            frame_height=frame_height,
            tracking_enabled=True,
            detections_count=detections_count,
            visible_tracks=[replace(track) for track in visible_tracks],
            held_tracks=[replace(track) for track in held_tracks],
            tentative_tracks=[replace(track) for track in tentative_tracks],
            lost_tracks=[replace(track) for track in lost_tracks],
            removed_track_ids=list(removed_track_ids or []),
        )

    def get_frame_tracking_tracks(self) -> list[Track]:
        return [
            track
            for track in self._iter_non_removed_tracks()
            if track.state in {TrackState.CONFIRMED, TrackState.HELD}
        ]

    def get_track(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def has_confirmed_or_held_tracks(self) -> bool:
        return any(track.state in {TrackState.CONFIRMED, TrackState.HELD} for track in self._tracks.values())

    def _run_detection_update(
        self,
        detections: list[Detection],
        frame_shape: FrameShape,
        frame_index: int,
        timestamp: float | None,
    ) -> FrameTrackingSummary:
        frame_height, frame_width = frame_shape[:2]
        now = timestamp if timestamp is not None else time.time()
        removed_track_ids: list[int] = []

        candidate_track_ids = self._candidate_track_ids()
        predicted_bboxes = {
            track_id: self._predict_bbox(self._tracks[track_id], frame_width, frame_height)
            for track_id in candidate_track_ids
        }
        for track_id, predicted_bbox in predicted_bboxes.items():
            self._tracks[track_id].predicted_bbox = predicted_bbox

        matches = self._greedy_match(
            detections=detections,
            candidate_track_ids=candidate_track_ids,
            predicted_bboxes=predicted_bboxes,
        )
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index in matches}

        for track_id, detection_index in matches:
            self._update_matched_track(
                track=self._tracks[track_id],
                detection=detections[detection_index],
                frame_width=frame_width,
                frame_height=frame_height,
                frame_index=frame_index,
                timestamp=now,
            )

        for track in self._iter_non_removed_tracks():
            if track.track_id in matched_track_ids:
                continue
            if self._handle_unmatched_track(
                track=track,
                frame_width=frame_width,
                frame_height=frame_height,
                timestamp=now,
            ):
                removed_track_ids.append(track.track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            if self._should_reserve_for_existing_track(
                detection=detection,
                frame_width=frame_width,
                frame_height=frame_height,
            ):
                continue
            if detection.confidence < self._config.acquire_confidence_threshold:
                continue
            if self._active_track_count() >= self._config.max_active_tracks:
                self._logger.warning(
                    "Skipping new detection because max_active_tracks=%d reached.",
                    self._config.max_active_tracks,
                )
                break
            self._create_track(
                detection=detection,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_index=frame_index,
                timestamp=now,
            )

        return self.build_summary(
            frame_shape=frame_shape,
            frame_index=frame_index,
            detections_count=len(detections),
            removed_track_ids=removed_track_ids,
        )

    def _candidate_track_ids(self) -> list[int]:
        candidate_ids: list[int] = []
        for track in self._iter_non_removed_tracks():
            if track.state in {TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.HELD}:
                candidate_ids.append(track.track_id)
                continue
            if self._can_reacquire_lost_track(track):
                candidate_ids.append(track.track_id)
        return candidate_ids

    def _greedy_match(
        self,
        detections: list[Detection],
        candidate_track_ids: list[int],
        predicted_bboxes: dict[int, FloatBBox],
    ) -> list[tuple[int, int]]:
        match_candidates: list[tuple[int, int, float, float, float, float, int, int]] = []
        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            predicted_bbox = predicted_bboxes[track_id]

            for detection_index, detection in enumerate(detections):
                quality_rank, iou, distance, ratio = self._match_quality(
                    track=track,
                    predicted_bbox=predicted_bbox,
                    detection=detection,
                )
                if quality_rank is None or iou is None or distance is None or ratio is None:
                    continue

                match_candidates.append(
                    (
                        self._state_priority(track.state),
                        quality_rank,
                        -iou,
                        distance,
                        abs(1.0 - ratio),
                        -detection.confidence,
                        track_id,
                        detection_index,
                    ),
                )

        match_candidates.sort()
        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, _, _, _, _, _, track_id, detection_index in match_candidates:
            if track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)
            matches.append((track_id, detection_index))
        return matches

    def _match_quality(
        self,
        track: Track,
        predicted_bbox: FloatBBox,
        detection: Detection,
    ) -> tuple[int | None, float | None, float | None, float | None]:
        iou, distance, ratio, inside_expanded = self._association_metrics(
            reference_bbox=predicted_bbox,
            detection=detection,
        )

        if track.state == TrackState.TENTATIVE:
            if detection.confidence < self._config.acquire_confidence_threshold:
                return None, None, None, None
            if not self._passes_strict_gates(iou, distance, ratio):
                return None, None, None, None
            return 0, iou, distance, ratio

        if detection.confidence < self._config.keep_confidence_threshold:
            return None, None, None, None

        strict_match = self._passes_strict_gates(iou, distance, ratio)
        soft_match = self._passes_soft_gates(iou, distance, ratio, inside_expanded)
        if not soft_match:
            return None, None, None, None
        if strict_match and detection.confidence >= self._config.acquire_confidence_threshold:
            return 0, iou, distance, ratio
        return 1, iou, distance, ratio

    def _passes_strict_gates(self, iou: float, distance: float, ratio: float) -> bool:
        return (
            iou >= self._config.iou_gate
            and distance <= self._config.center_distance_gate
            and self._config.min_area_ratio <= ratio <= self._config.max_area_ratio
        )

    def _passes_soft_gates(
        self,
        iou: float,
        distance: float,
        ratio: float,
        inside_expanded: bool,
    ) -> bool:
        return (
            (
                iou >= self._config.soft_iou_gate
                or distance <= self._config.soft_center_distance_gate
                or inside_expanded
            )
            and self._config.soft_min_area_ratio <= ratio <= self._config.soft_max_area_ratio
        )

    def _update_matched_track(
        self,
        track: Track,
        detection: Detection,
        frame_width: int,
        frame_height: int,
        frame_index: int,
        timestamp: float,
    ) -> None:
        previous_state = track.state
        previous_center_x = track.center_x
        previous_center_y = track.center_y
        reference_bbox = track.predicted_bbox or track.bbox
        detection_bbox = clamp_float_bbox(detection.bbox, frame_width, frame_height)
        smoothed_bbox = smooth_bbox(reference_bbox, detection_bbox, self._config.smoothing_alpha)
        smoothed_bbox = clamp_float_bbox(smoothed_bbox, frame_width, frame_height)

        self._update_track_geometry(
            track=track,
            bbox=smoothed_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        track.velocity_x = smooth_value(
            track.velocity_x,
            detection.center_x - previous_center_x,
            self._VELOCITY_ALPHA,
        )
        track.velocity_y = smooth_value(
            track.velocity_y,
            detection.center_y - previous_center_y,
            self._VELOCITY_ALPHA,
        )
        track.confidence = detection.confidence
        track.age += 1
        track.hits += 1
        track.consecutive_hits += 1
        track.consecutive_misses = 0
        track.last_seen_frame = frame_index
        track.last_detection_frame = frame_index
        track.last_update_ts = timestamp
        track.predicted_bbox = None
        track.held_frames = 0
        track.tracker_only_updates = 0
        track.tracker_failures = 0

        if previous_state == TrackState.TENTATIVE:
            if (
                track.consecutive_hits >= self._config.confirm_frames
                and detection.confidence >= self._config.acquire_confidence_threshold
            ):
                track.state = TrackState.CONFIRMED
                track.reconfirm_hits = 0
                self._logger.info("track_confirmed track_id=%d", track.track_id)
        elif previous_state == TrackState.CONFIRMED:
            track.state = TrackState.CONFIRMED
            track.reconfirm_hits = 0
        elif previous_state == TrackState.HELD:
            track.state = TrackState.HELD
            track.reconfirm_hits += 1
            if track.reconfirm_hits >= self._config.reconfirm_frames:
                track.state = TrackState.CONFIRMED
                track.reconfirm_hits = 0
                self._logger.info("track_reacquired track_id=%d", track.track_id)
        elif previous_state == TrackState.LOST:
            track.state = TrackState.HELD
            track.reconfirm_hits = 1
            if track.reconfirm_hits >= self._config.reconfirm_frames:
                track.state = TrackState.CONFIRMED
                track.reconfirm_hits = 0
                self._logger.info("track_reacquired track_id=%d", track.track_id)

        self._logger.info("track_updated track_id=%d", track.track_id)

    def _handle_unmatched_track(
        self,
        track: Track,
        frame_width: int,
        frame_height: int,
        timestamp: float,
    ) -> bool:
        track.age += 1
        track.misses += 1
        track.consecutive_hits = 0
        track.consecutive_misses += 1
        track.last_update_ts = timestamp
        track.display_number = None
        track.reconfirm_hits = 0

        if track.state == TrackState.TENTATIVE:
            track.state = TrackState.REMOVED
            self._logger.info("track_removed track_id=%d", track.track_id)
            return True

        self._apply_prediction(track, frame_width, frame_height)

        if track.state == TrackState.CONFIRMED:
            if (
                self._config.preserve_confirmed_tracks
                or self._config.never_drop_confirmed_on_single_bad_frame
            ):
                track.state = TrackState.HELD
                track.held_frames = 1
            else:
                track.state = TrackState.LOST
                self._logger.info("track_lost track_id=%d", track.track_id)
        elif track.state == TrackState.HELD:
            track.held_frames += 1
            if track.consecutive_misses >= self._lost_transition_threshold():
                track.state = TrackState.LOST
                self._logger.info("track_lost track_id=%d", track.track_id)
        elif track.state == TrackState.LOST:
            track.held_frames = 0

        if track.consecutive_misses >= self._config.hard_remove_frames:
            track.state = TrackState.REMOVED
            track.display_number = None
            self._logger.info("track_removed track_id=%d", track.track_id)
            return True

        return False

    def _create_track(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int,
        frame_index: int,
        timestamp: float,
    ) -> Track:
        bbox = clamp_float_bbox(detection.bbox, frame_width, frame_height)
        center_x, center_y = calculate_center_from_bbox(bbox)
        normalized_x, normalized_y = normalize_coordinates(
            center_x,
            center_y,
            frame_width,
            frame_height,
        )
        track_id = self._next_track_id
        self._next_track_id += 1

        state = (
            TrackState.CONFIRMED
            if self._config.confirm_frames <= 1
            and detection.confidence >= self._config.acquire_confidence_threshold
            else TrackState.TENTATIVE
        )
        track = Track(
            track_id=track_id,
            display_number=None,
            state=state,
            bbox=bbox,
            square_bbox=bbox_to_square(
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                frame_width,
                frame_height,
            ),
            center_x=center_x,
            center_y=center_y,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
            confidence=detection.confidence,
            age=1,
            hits=1,
            misses=0,
            consecutive_hits=1,
            consecutive_misses=0,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            last_detection_frame=frame_index,
            last_update_ts=timestamp,
            reconfirm_hits=0,
            held_frames=0,
            tracker_only_updates=0,
            tracker_failures=0,
            class_name=detection.class_name,
        )
        self._tracks[track_id] = track
        self._logger.info("track_created track_id=%d", track.track_id)
        if track.state == TrackState.CONFIRMED:
            self._logger.info("track_confirmed track_id=%d", track.track_id)
        return track

    def _should_reserve_for_existing_track(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        for track in self._tracks.values():
            if not self._is_reservable_track(track):
                continue
            reference_bbox = self._predict_bbox(track, frame_width, frame_height)
            iou, distance, ratio, inside_expanded = self._association_metrics(
                reference_bbox=reference_bbox,
                detection=detection,
            )
            if self._passes_duplicate_reserve_gates(
                reference_bbox=reference_bbox,
                iou=iou,
                distance=distance,
                ratio=ratio,
                inside_expanded=inside_expanded,
            ):
                return True
        return False

    def _association_metrics(
        self,
        reference_bbox: FloatBBox,
        detection: Detection,
    ) -> tuple[float, float, float, bool]:
        reference_center = calculate_center_from_bbox(reference_bbox)
        reference_area = bbox_area(reference_bbox)
        iou = bbox_iou(reference_bbox, detection.bbox)
        distance = center_distance(reference_center, (detection.center_x, detection.center_y))
        ratio = area_ratio(detection.area, reference_area)
        inside_expanded = self._is_inside_expanded_window(
            reference_bbox,
            (detection.center_x, detection.center_y),
        )
        return iou, distance, ratio, inside_expanded

    def _is_reservable_track(self, track: Track) -> bool:
        if track.state in {TrackState.CONFIRMED, TrackState.HELD}:
            return True
        return self._can_reacquire_lost_track(track)

    def _passes_duplicate_reserve_gates(
        self,
        reference_bbox: FloatBBox,
        iou: float,
        distance: float,
        ratio: float,
        inside_expanded: bool,
    ) -> bool:
        width = max(1.0, reference_bbox[2] - reference_bbox[0])
        height = max(1.0, reference_bbox[3] - reference_bbox[1])
        duplicate_distance_gate = min(
            self._config.soft_center_distance_gate,
            max(width, height) * 3.0,
        )
        return (
            self._config.soft_min_area_ratio <= ratio <= self._config.soft_max_area_ratio
            and (
                iou >= self._config.soft_iou_gate
                or inside_expanded
                or distance <= duplicate_distance_gate
            )
        )

    def _can_reacquire_lost_track(self, track: Track) -> bool:
        return (
            track.state == TrackState.LOST
            and self._config.keep_lost_tracks
            and self._lost_frames(track) <= self._config.reacquire_max_frames
        )

    def _lost_frames(self, track: Track) -> int:
        if track.state != TrackState.LOST:
            return 0
        return max(1, track.consecutive_misses - self._lost_transition_threshold() + 1)

    def _assign_display_numbers(self, frame_index: int) -> list[Track]:
        for track in self._tracks.values():
            if not track.is_displayed(frame_index):
                track.display_number = None

        visible_tracks = sort_tracks_for_display(
            [
                track
                for track in self._iter_non_removed_tracks()
                if track.is_displayed(frame_index)
            ],
            mode=self._config.display_sort_mode,
        )
        for display_number, track in enumerate(visible_tracks, start=1):
            track.display_number = display_number
        return visible_tracks

    def _apply_prediction(
        self,
        track: Track,
        frame_width: int,
        frame_height: int,
    ) -> None:
        predicted_bbox = track.predicted_bbox or self._predict_bbox(track, frame_width, frame_height)
        self._update_track_geometry(
            track=track,
            bbox=predicted_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        track.predicted_bbox = None

    def _predict_bbox(
        self,
        track: Track,
        frame_width: int,
        frame_height: int,
    ) -> FloatBBox:
        if not self._config.use_motion_prediction:
            return track.bbox
        if track.state == TrackState.TENTATIVE:
            return track.bbox
        predicted_bbox = (
            track.bbox[0] + track.velocity_x,
            track.bbox[1] + track.velocity_y,
            track.bbox[2] + track.velocity_x,
            track.bbox[3] + track.velocity_y,
        )
        return clamp_float_bbox(predicted_bbox, frame_width, frame_height)

    def _update_track_geometry(
        self,
        track: Track,
        bbox: FloatBBox,
        frame_width: int,
        frame_height: int,
    ) -> None:
        track.bbox = clamp_float_bbox(bbox, frame_width, frame_height)
        track.square_bbox = bbox_to_square(
            track.bbox[0],
            track.bbox[1],
            track.bbox[2],
            track.bbox[3],
            frame_width,
            frame_height,
        )
        track.center_x, track.center_y = calculate_center_from_bbox(track.bbox)
        track.normalized_x, track.normalized_y = normalize_coordinates(
            track.center_x,
            track.center_y,
            frame_width,
            frame_height,
        )

    def _is_inside_expanded_window(
        self,
        bbox: FloatBBox,
        center: tuple[float, float],
    ) -> bool:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        margin_x = max(width * 0.5, 8.0)
        margin_y = max(height * 0.5, 8.0)
        return (
            (bbox[0] - margin_x) <= center[0] <= (bbox[2] + margin_x)
            and (bbox[1] - margin_y) <= center[1] <= (bbox[3] + margin_y)
        )

    def _lost_transition_threshold(self) -> int:
        return max(
            self._config.max_missing_frames,
            self._config.hold_without_detection_frames + self._config.lost_transition_frames,
        )

    def _iter_non_removed_tracks(self) -> list[Track]:
        return [track for track in self._tracks.values() if track.state != TrackState.REMOVED]

    def _active_track_count(self) -> int:
        return sum(1 for track in self._tracks.values() if track.is_active)

    @staticmethod
    def _state_priority(state: TrackState) -> int:
        priorities = {
            TrackState.CONFIRMED: 0,
            TrackState.HELD: 1,
            TrackState.LOST: 2,
            TrackState.TENTATIVE: 3,
            TrackState.REMOVED: 4,
        }
        return priorities[state]


class DetectThenTrackManager:
    def __init__(
        self,
        config: TrackingConfig,
        logger: logging.Logger,
        frame_tracker_factory: Callable[[], FrameTrackerWrapper] | None = None,
    ) -> None:
        if not config.tracking_enabled:
            raise ValueError("DetectThenTrackManager requires tracking_enabled=true.")
        self._config = config
        self._logger = logger
        self._track_manager = MultiCatTracker(config=config, logger=logger)
        self._frame_tracker_factory = frame_tracker_factory or OpenCvFrameTracker
        self._frame_trackers: dict[int, FrameTrackerWrapper] = {}
        self._last_detector_frame = 0
        self._had_confirmed_track = False
        self._pipeline_state = TrackingPipelineState.SEARCH

    @property
    def pipeline_state(self) -> TrackingPipelineState:
        return self._pipeline_state

    def should_run_detector(self, frame_index: int, detection_cycle_due: bool) -> bool:
        if not self._config.tracker_only_mode_after_confirm:
            return detection_cycle_due

        frame_tracking_tracks = self._track_manager.get_frame_tracking_tracks()
        if not frame_tracking_tracks:
            if self._pipeline_state == TrackingPipelineState.REACQUIRE:
                return True
            return detection_cycle_due

        if any(
            track.tracker_failures >= self._config.reacquire_after_failed_tracker_frames
            for track in frame_tracking_tracks
        ):
            return True
        if any(
            (frame_index - track.last_detection_frame) >= self._config.max_tracker_only_frames_without_detection
            for track in frame_tracking_tracks
        ):
            return True
        if (frame_index - self._last_detector_frame) >= self._config.detector_interval_while_tracking:
            return True
        return False

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float | None = None,
        detections: list[Detection] | None = None,
    ) -> FrameTrackingSummary:
        now = timestamp if timestamp is not None else time.time()
        frame_shape = frame.shape
        removed_track_ids: list[int] = []
        tracker_updates_count = 0
        tracker_failures_count = 0
        yolo_ran_this_frame = detections is not None

        if self._config.tracker_only_mode_after_confirm:
            self._prune_frame_trackers()
            if not yolo_ran_this_frame:
                tracker_updates_count, tracker_failures_count, tracker_removed = self._update_frame_trackers(
                    frame=frame,
                    frame_shape=frame_shape,
                    frame_index=frame_index,
                    timestamp=now,
                )
                removed_track_ids.extend(tracker_removed)

        if yolo_ran_this_frame:
            summary = self._track_manager.update(
                detections=detections,
                frame_shape=frame_shape,
                frame_index=frame_index,
                timestamp=now,
            )
            self._last_detector_frame = frame_index
        else:
            summary = self._track_manager.build_summary(
                frame_shape=frame_shape,
                frame_index=frame_index,
                detections_count=0,
                removed_track_ids=removed_track_ids,
            )

        if removed_track_ids:
            summary.removed_track_ids.extend(track_id for track_id in removed_track_ids if track_id not in summary.removed_track_ids)

        if self._config.tracker_only_mode_after_confirm:
            self._sync_frame_trackers(frame=frame, frame_index=frame_index)

        if summary.confirmed_count > 0 or summary.held_count > 0:
            self._had_confirmed_track = True
        self._pipeline_state = self._determine_pipeline_state(frame_index)

        summary.pipeline_state = self._pipeline_state
        summary.yolo_ran_this_frame = yolo_ran_this_frame
        summary.tracker_updates_count = tracker_updates_count
        summary.tracker_failures_count = tracker_failures_count
        return summary

    def _update_frame_trackers(
        self,
        frame: np.ndarray,
        frame_shape: FrameShape,
        frame_index: int,
        timestamp: float,
    ) -> tuple[int, int, list[int]]:
        updates = 0
        failures = 0
        removed_track_ids: list[int] = []
        for track_id, tracker in list(self._frame_trackers.items()):
            track = self._track_manager.get_track(track_id)
            if track is None or track.state not in {TrackState.CONFIRMED, TrackState.HELD}:
                self._frame_trackers.pop(track_id, None)
                continue

            ok, bbox = tracker.update(frame)
            if ok and self._is_valid_bbox(bbox, frame_shape):
                self._track_manager.update_from_tracker(
                    track_id=track_id,
                    bbox=bbox,
                    frame_shape=frame_shape,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
                updates += 1
                continue

            failures += 1
            was_removed = self._track_manager.mark_tracker_failure(
                track_id=track_id,
                frame_shape=frame_shape,
                timestamp=timestamp,
            )
            if was_removed:
                removed_track_ids.append(track_id)
            if self._track_manager.get_track(track_id) is None:
                self._frame_trackers.pop(track_id, None)
                continue

            updated_track = self._track_manager.get_track(track_id)
            if updated_track is None or updated_track.state in {TrackState.LOST, TrackState.REMOVED}:
                self._frame_trackers.pop(track_id, None)
        return updates, failures, removed_track_ids

    def _sync_frame_trackers(self, frame: np.ndarray, frame_index: int) -> None:
        active_track_ids = {
            track.track_id
            for track in self._track_manager.get_frame_tracking_tracks()
        }
        for track_id in list(self._frame_trackers):
            if track_id not in active_track_ids:
                self._frame_trackers.pop(track_id, None)

        for track in self._track_manager.get_frame_tracking_tracks():
            needs_reinitialize = (
                track.track_id not in self._frame_trackers
                or track.last_detection_frame == frame_index
            )
            if not needs_reinitialize:
                continue
            frame_tracker = self._frame_tracker_factory()
            try:
                initialized = frame_tracker.initialize(frame, track.bbox)
            except Exception:
                initialized = False

            if initialized:
                self._frame_trackers[track.track_id] = frame_tracker
            else:
                self._logger.warning(
                    "Frame tracker init failed for track_id=%d bbox=%s",
                    track.track_id,
                    track.bbox,
                )

    def _prune_frame_trackers(self) -> None:
        active_track_ids = {
            track.track_id
            for track in self._track_manager.get_frame_tracking_tracks()
        }
        for track_id in list(self._frame_trackers):
            if track_id not in active_track_ids:
                self._frame_trackers.pop(track_id, None)

    def _determine_pipeline_state(self, frame_index: int) -> TrackingPipelineState:
        frame_tracking_tracks = self._track_manager.get_frame_tracking_tracks()
        if frame_tracking_tracks and self._config.tracker_only_mode_after_confirm:
            if any(
                track.tracker_failures >= self._config.reacquire_after_failed_tracker_frames
                or (frame_index - track.last_detection_frame) >= self._config.max_tracker_only_frames_without_detection
                for track in frame_tracking_tracks
            ):
                return TrackingPipelineState.REACQUIRE
            return TrackingPipelineState.TRACK_ONLY

        if any(
            self._track_manager._can_reacquire_lost_track(track)
            for track in self._track_manager._iter_non_removed_tracks()
        ):
            return TrackingPipelineState.REACQUIRE
        if self._had_confirmed_track:
            return TrackingPipelineState.LOST
        return TrackingPipelineState.SEARCH

    @staticmethod
    def _is_valid_bbox(bbox: FloatBBox, frame_shape: FrameShape) -> bool:
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width < 4.0 or height < 4.0:
            return False
        if x2 <= 0 or y2 <= 0:
            return False
        if x1 >= frame_width or y1 >= frame_height:
            return False
        return True
