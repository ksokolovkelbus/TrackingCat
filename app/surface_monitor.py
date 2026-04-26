from __future__ import annotations

import logging
from dataclasses import dataclass

from app.audio_alert import AudioAlertPlayer
from app.models import (
    ActiveAlertTrack,
    FrameShape,
    SurfaceAlertConfig,
    SurfaceEvent,
    Track,
    TrackLocationState,
    TrackState,
    ZoneType,
)
from app.zones import SceneZoneClassifier


@dataclass(frozen=True, slots=True)
class SurfaceMonitorResult:
    track_location_states: dict[int, TrackLocationState]
    surface_events: list[SurfaceEvent]
    suppressed_surface_events: list[SurfaceEvent]
    active_alert_tracks: list[ActiveAlertTrack]
    overlay_message: str | None


class SurfaceMonitor:
    def __init__(
        self,
        classifier: SceneZoneClassifier,
        config: SurfaceAlertConfig,
        logger: logging.Logger,
        audio_player: AudioAlertPlayer | None = None,
    ) -> None:
        self._classifier = classifier
        self._config = config
        self._logger = logger
        self._audio_player = audio_player or AudioAlertPlayer(config=config, logger=logger)
        self._track_states: dict[int, TrackLocationState] = {}
        self._last_global_sound_ts = 0.0
        self._overlay_message: str | None = None
        self._overlay_message_until_frame = 0

    def cleanup_removed_tracks(self, removed_track_ids: list[int]) -> None:
        for track_id in removed_track_ids:
            self._track_states.pop(track_id, None)

        if self._config.continuous_while_in_zone and self._config.stop_when_zone_empty and not self._track_states:
            self._audio_player.stop_continuous()

    def update(
        self,
        tracks: list[Track],
        frame_shape: FrameShape,
        frame_index: int,
        timestamp: float,
    ) -> SurfaceMonitorResult:
        location_states: dict[int, TrackLocationState] = {}
        emitted_events: list[SurfaceEvent] = []
        suppressed_events: list[SurfaceEvent] = []

        for track in tracks:
            location_state = self._track_states.get(track.track_id)
            if location_state is None:
                location_state = TrackLocationState(track_id=track.track_id)
                self._track_states[track.track_id] = location_state

            if self._is_alert_eligible_track(track):
                zone_match = self._classifier.classify_alert_track(
                    track=track,
                    frame_shape=frame_shape,
                    point_mode=self._config.alert_point_mode,
                )
                zone_type = zone_match.zone.zone_type if zone_match.zone is not None else None
                zone_name = zone_match.zone.name if zone_match.zone is not None else None
            else:
                zone_type = None
                zone_name = None

            self._update_track_zone_state(
                track=track,
                state=location_state,
                zone_type=zone_type,
                zone_name=zone_name,
                frame_index=frame_index,
                timestamp=timestamp,
            )

            surface_event = self._maybe_build_surface_event(
                track=track,
                state=location_state,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            if surface_event is not None:
                if self._handle_surface_event(surface_event, location_state, frame_index, timestamp):
                    emitted_events.append(surface_event)
                else:
                    suppressed_events.append(surface_event)

            location_states[track.track_id] = TrackLocationState(
                track_id=location_state.track_id,
                current_zone_name=location_state.current_zone_name,
                current_zone_type=location_state.current_zone_type,
                previous_zone_name=location_state.previous_zone_name,
                previous_zone_type=location_state.previous_zone_type,
                is_on_surface=location_state.is_on_surface,
                last_surface_event_ts=location_state.last_surface_event_ts,
                last_sound_ts=location_state.last_sound_ts,
                last_zone_change_ts=location_state.last_zone_change_ts,
                last_frame_index=location_state.last_frame_index,
                last_alert_zone_name=location_state.last_alert_zone_name,
                alert_entered_ts=location_state.alert_entered_ts,
                current_zone_frames=location_state.current_zone_frames,
            )

        active_alert_tracks = self._build_active_alert_tracks(
            tracks=tracks,
            location_states=location_states,
            timestamp=timestamp,
        )
        self._sync_continuous_alert(
            active_alert_tracks=active_alert_tracks,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        overlay_message = (
            self._overlay_message
            if self._overlay_message and frame_index <= self._overlay_message_until_frame
            else None
        )
        if overlay_message is None:
            self._overlay_message = None
            self._overlay_message_until_frame = 0

        return SurfaceMonitorResult(
            track_location_states=location_states,
            surface_events=emitted_events,
            suppressed_surface_events=suppressed_events,
            active_alert_tracks=active_alert_tracks,
            overlay_message=overlay_message,
        )

    def close(self) -> None:
        self._audio_player.close()

    def _build_active_alert_tracks(
        self,
        tracks: list[Track],
        location_states: dict[int, TrackLocationState],
        timestamp: float,
    ) -> list[ActiveAlertTrack]:
        display_numbers_by_track_id = {track.track_id: track.display_number for track in tracks}
        active_tracks: list[ActiveAlertTrack] = []
        for track_id, state in location_states.items():
            if not self._is_surface_like(state.current_zone_type) or not state.current_zone_name:
                continue
            entered_at_ts = state.alert_entered_ts if state.alert_entered_ts > 0 else timestamp
            active_tracks.append(
                ActiveAlertTrack(
                    track_id=track_id,
                    display_number=display_numbers_by_track_id.get(track_id),
                    zone_name=state.current_zone_name,
                    zone_type=state.current_zone_type or ZoneType.SURFACE,
                    entered_at_ts=entered_at_ts,
                    dwell_seconds=max(0.0, timestamp - entered_at_ts),
                ),
            )
        active_tracks.sort(key=lambda item: (item.display_number is None, item.display_number or item.track_id))
        return active_tracks

    def _sync_continuous_alert(
        self,
        active_alert_tracks: list[ActiveAlertTrack],
        frame_index: int,
        timestamp: float,
    ) -> None:
        if not self._config.enabled or not self._config.continuous_while_in_zone:
            if self._config.stop_when_zone_empty:
                self._audio_player.stop_continuous()
            return

        active_zone_types = set(self._config.continuous_zone_types)
        matching_tracks = [
            track
            for track in active_alert_tracks
            if track.zone_type.value in active_zone_types
        ]

        if not matching_tracks:
            if self._config.stop_when_zone_empty:
                self._audio_player.stop_continuous()
            return

        primary_track = matching_tracks[0]
        event = SurfaceEvent(
            event_type="entered_surface",
            track_id=primary_track.track_id,
            zone_name=primary_track.zone_name,
            zone_type=primary_track.zone_type,
            frame_index=frame_index,
            timestamp=timestamp,
            message=f"ALERT: Cat on {primary_track.zone_name}",
        )

        self._audio_player.start_continuous(event)

        if self._config.show_overlay_message and self._config.overlay_message_frames > 0:
            self._overlay_message = event.message
            self._overlay_message_until_frame = frame_index + self._config.overlay_message_frames - 1

    def _update_track_zone_state(
        self,
        track: Track,
        state: TrackLocationState,
        zone_type: ZoneType | None,
        zone_name: str | None,
        frame_index: int,
        timestamp: float,
    ) -> None:
        previous_zone_name = state.current_zone_name
        previous_zone_type = state.current_zone_type
        state.previous_zone_name = previous_zone_name
        state.previous_zone_type = previous_zone_type

        zone_changed = previous_zone_name != zone_name or previous_zone_type != zone_type
        state.previous_floor_frames = state.floor_frames
        if zone_changed:
            state.current_zone_name = zone_name
            state.current_zone_type = zone_type
            state.last_zone_change_ts = timestamp
            state.current_zone_frames = 1 if zone_type is not None and zone_name else 0

            if zone_type is not None and zone_name:
                self._logger.info(
                    "track_id=%d entered %s zone %s",
                    track.track_id,
                    zone_type.value,
                    zone_name,
                )
            elif previous_zone_type is not None and previous_zone_name is not None:
                self._logger.info(
                    "track_id=%d left %s zone %s",
                    track.track_id,
                    previous_zone_type.value,
                    previous_zone_name,
                )
        else:
            state.current_zone_name = zone_name
            state.current_zone_type = zone_type
            if zone_type is not None and zone_name:
                state.current_zone_frames += 1
            else:
                state.current_zone_frames = 0

        if zone_type == ZoneType.FLOOR:
            if previous_zone_type == ZoneType.FLOOR and not zone_changed:
                state.floor_frames += 1
            else:
                state.floor_frames = 1
        else:
            state.floor_frames = 0

        if self._is_surface_like(zone_type):
            if not self._is_surface_like(previous_zone_type) or previous_zone_name != zone_name:
                state.alert_entered_ts = timestamp
        elif self._is_surface_like(previous_zone_type):
            state.alert_entered_ts = 0.0

        state.is_on_surface = self._is_surface_like(zone_type)
        state.last_frame_index = frame_index

        if zone_type == ZoneType.FLOOR and previous_zone_type in {ZoneType.SURFACE, ZoneType.RESTRICTED} and zone_name:
            self._logger.info(
                "track_id=%d returned to floor zone %s",
                track.track_id,
                zone_name,
            )

    def _maybe_build_surface_event(
        self,
        track: Track,
        state: TrackLocationState,
        frame_index: int,
        timestamp: float,
    ) -> SurfaceEvent | None:
        if not self._config.enabled or not self._config.trigger_on_surface_entry:
            return None
        if not self._is_surface_like(state.current_zone_type) or not state.current_zone_name:
            return None

        previous_zone_type = state.previous_zone_type
        delayed_same_zone_entry = False
        if self._is_surface_like(previous_zone_type):
            delayed_same_zone_entry = (
                previous_zone_type == state.current_zone_type
                and state.previous_zone_name == state.current_zone_name
                and state.last_alert_zone_name != state.current_zone_name
                and state.current_zone_frames == self._config.min_zone_frames_before_alert
            )
            if not delayed_same_zone_entry:
                return None
        elif previous_zone_type is None:
            if not self._config.trigger_from_unknown:
                return None
        elif self._config.trigger_only_from_floor and previous_zone_type != ZoneType.FLOOR:
            return None

        if (
            previous_zone_type == ZoneType.FLOOR
            and state.previous_floor_frames < self._config.min_floor_frames_before_alert
        ):
            return None

        if state.current_zone_frames < self._config.min_zone_frames_before_alert:
            return None

        if (
            not self._config.repeat_on_same_surface
            and state.last_alert_zone_name == state.current_zone_name
            and previous_zone_type != ZoneType.FLOOR
            and not delayed_same_zone_entry
        ):
            return None

        return SurfaceEvent(
            event_type="entered_surface",
            track_id=track.track_id,
            zone_name=state.current_zone_name,
            zone_type=state.current_zone_type or ZoneType.SURFACE,
            frame_index=frame_index,
            timestamp=timestamp,
            message=f"ALERT: Cat on {state.current_zone_name}",
        )

    def _handle_surface_event(
        self,
        event: SurfaceEvent,
        state: TrackLocationState,
        frame_index: int,
        timestamp: float,
    ) -> bool:
        suppressed_reason = self._suppression_reason(state, timestamp)
        if suppressed_reason is not None:
            state.last_alert_zone_name = event.zone_name
            self._logger.info(
                "surface_alert_suppressed_by_%s track_id=%d zone=%s",
                suppressed_reason,
                event.track_id,
                event.zone_name,
            )
            return False

        self._logger.info(
            "surface_alert_emitted track_id=%d zone=%s",
            event.track_id,
            event.zone_name,
        )
        playback_result = self._audio_player.play(event)
        state.last_surface_event_ts = timestamp
        state.last_alert_zone_name = event.zone_name
        if playback_result.played:
            state.last_sound_ts = timestamp
            self._last_global_sound_ts = timestamp

        if self._config.show_overlay_message and self._config.overlay_message_frames > 0:
            self._overlay_message = event.message
            self._overlay_message_until_frame = frame_index + self._config.overlay_message_frames - 1
        return True

    def _suppression_reason(
        self,
        state: TrackLocationState,
        timestamp: float,
    ) -> str | None:
        if state.last_surface_event_ts > 0 and (timestamp - state.last_surface_event_ts) < self._config.cooldown_seconds:
            return "cooldown"
        if state.last_sound_ts > 0 and (timestamp - state.last_sound_ts) < self._config.min_interval_per_track:
            return "track_interval"
        if self._last_global_sound_ts > 0 and (timestamp - self._last_global_sound_ts) < self._config.global_min_interval:
            return "global_interval"
        return None

    @staticmethod
    def _is_surface_like(zone_type: ZoneType | None) -> bool:
        return zone_type in {ZoneType.SURFACE, ZoneType.RESTRICTED}

    @staticmethod
    def _is_alert_eligible_track(track: Track) -> bool:
        return track.state == TrackState.CONFIRMED
