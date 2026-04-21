import logging
from unittest.mock import patch

import cv2
import numpy as np

from app.models import SourceConfig
from app.video_source import VideoSource, VideoSourceError


class _DummyResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_jpeg_bytes(width: int = 32, height: int = 24) -> bytes:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok is True
    return encoded.tobytes()


def test_http_snapshot_source_reads_frame() -> None:
    config = SourceConfig(
        source_type='http_snapshot',
        stream_url='http://192.168.8.140/snapshot.jpg',
        snapshot_timeout_seconds=1.0,
    )
    source = VideoSource(config=config, logger=logging.getLogger('test_snapshot'))

    with patch('app.video_source.urllib_request.urlopen', return_value=_DummyResponse(_make_jpeg_bytes())):
        source.open()
        ok, frame = source.read()

    assert ok is True
    assert frame is not None
    assert frame.shape[:2] == (24, 32)


def test_http_snapshot_source_open_fails_on_invalid_payload() -> None:
    config = SourceConfig(
        source_type='http_snapshot',
        stream_url='http://192.168.8.140/snapshot.jpg',
        snapshot_timeout_seconds=1.0,
    )
    source = VideoSource(config=config, logger=logging.getLogger('test_snapshot'))

    with patch('app.video_source.urllib_request.urlopen', return_value=_DummyResponse(b'not-a-jpeg')):
        try:
            source.open()
        except VideoSourceError:
            pass
        else:
            raise AssertionError('Expected VideoSourceError for invalid snapshot payload')
