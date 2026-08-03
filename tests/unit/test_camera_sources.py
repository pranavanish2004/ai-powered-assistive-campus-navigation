"""Tests for IPCameraSource and VideoFileCamera, mocking cv2.VideoCapture
since no real network camera or physical hardware is available in CI --
consistent with how OpenCVCamera's own hardware dependency is isolated."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from campus_nav.core.exceptions import CameraError
from campus_nav.infrastructure.camera.ip_camera import IPCameraSource
from campus_nav.infrastructure.camera.video_file_camera import VideoFileCamera

pytestmark = pytest.mark.unit


# --- IPCameraSource ---


def test_ip_camera_requires_non_empty_url():
    with pytest.raises(CameraError):
        IPCameraSource(stream_url="")


@patch("campus_nav.infrastructure.camera.ip_camera.cv2.VideoCapture")
def test_ip_camera_start_success(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_video_capture.return_value = mock_capture

    camera = IPCameraSource(stream_url="http://192.168.1.5:4747/video")
    camera.start()

    assert camera.is_available() is True
    mock_video_capture.assert_called_once_with("http://192.168.1.5:4747/video")


@patch("campus_nav.infrastructure.camera.ip_camera.cv2.VideoCapture")
def test_ip_camera_start_failure_raises_camera_error(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_video_capture.return_value = mock_capture

    camera = IPCameraSource(stream_url="http://unreachable/video")
    with pytest.raises(CameraError):
        camera.start()


@patch("campus_nav.infrastructure.camera.ip_camera.cv2.VideoCapture")
def test_ip_camera_get_frame_returns_frame_on_success(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, frame)
    mock_video_capture.return_value = mock_capture

    camera = IPCameraSource(stream_url="http://example/video")
    camera.start()
    result = camera.get_frame()

    assert result is frame


@patch("campus_nav.infrastructure.camera.ip_camera.cv2.VideoCapture")
def test_ip_camera_get_frame_returns_none_on_read_failure(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (False, None)
    mock_video_capture.return_value = mock_capture

    camera = IPCameraSource(stream_url="http://example/video")
    camera.start()
    assert camera.get_frame() is None


def test_ip_camera_get_frame_before_start_returns_none():
    camera = IPCameraSource(stream_url="http://example/video")
    assert camera.get_frame() is None


@patch("campus_nav.infrastructure.camera.ip_camera.cv2.VideoCapture")
def test_ip_camera_stop_releases_capture(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_video_capture.return_value = mock_capture

    camera = IPCameraSource(stream_url="http://example/video")
    camera.start()
    camera.stop()

    mock_capture.release.assert_called_once()
    assert camera.is_available() is False


def test_ip_camera_stop_before_start_is_safe():
    camera = IPCameraSource(stream_url="http://example/video")
    camera.stop()  # should not raise


# --- VideoFileCamera ---


@patch("campus_nav.infrastructure.camera.video_file_camera.cv2.VideoCapture")
def test_video_file_camera_start_failure_raises(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_video_capture.return_value = mock_capture

    camera = VideoFileCamera(file_path="/nonexistent/video.mp4")
    with pytest.raises(CameraError):
        camera.start()


@patch("campus_nav.infrastructure.camera.video_file_camera.cv2.VideoCapture")
def test_video_file_camera_reads_frames(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, frame)
    mock_video_capture.return_value = mock_capture

    camera = VideoFileCamera(file_path="video.mp4")
    camera.start()
    assert camera.get_frame() is frame


@patch("campus_nav.infrastructure.camera.video_file_camera.cv2.VideoCapture")
def test_video_file_camera_loops_at_end_of_file(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    # First read fails (end of file), second read (after rewind) succeeds.
    mock_capture.read.side_effect = [(False, None), (True, frame)]
    mock_video_capture.return_value = mock_capture

    camera = VideoFileCamera(file_path="video.mp4", loop=True)
    camera.start()
    result = camera.get_frame()

    assert result is frame
    mock_capture.set.assert_called_once()  # rewound via CAP_PROP_POS_FRAMES


@patch("campus_nav.infrastructure.camera.video_file_camera.cv2.VideoCapture")
def test_video_file_camera_no_loop_returns_none_at_end(mock_video_capture):
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (False, None)
    mock_video_capture.return_value = mock_capture

    camera = VideoFileCamera(file_path="video.mp4", loop=False)
    camera.start()
    assert camera.get_frame() is None
    mock_capture.set.assert_not_called()
