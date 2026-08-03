"""Tests for the camera source factory (Settings-driven backend selection)."""

import pytest

from campus_nav.config.settings import Settings
from campus_nav.core.exceptions import ConfigurationError
from campus_nav.infrastructure.camera.camera_factory import create_camera
from campus_nav.infrastructure.camera.ip_camera import IPCameraSource
from campus_nav.infrastructure.camera.opencv_camera import OpenCVCamera
from campus_nav.infrastructure.camera.video_file_camera import VideoFileCamera

pytestmark = pytest.mark.unit


def test_usb_source_returns_opencv_camera():
    settings = Settings(_env_file=None, camera_source_type="usb")
    camera = create_camera(settings)
    assert isinstance(camera, OpenCVCamera)


def test_ip_source_returns_ip_camera():
    settings = Settings(_env_file=None, camera_source_type="ip", camera_ip_url="http://x/video")
    camera = create_camera(settings)
    assert isinstance(camera, IPCameraSource)


def test_ip_source_without_url_raises_configuration_error():
    settings = Settings(_env_file=None, camera_source_type="ip", camera_ip_url=None)
    with pytest.raises(ConfigurationError):
        create_camera(settings)


def test_file_source_returns_video_file_camera():
    settings = Settings(
        _env_file=None, camera_source_type="file", camera_video_file_path="video.mp4"
    )
    camera = create_camera(settings)
    assert isinstance(camera, VideoFileCamera)


def test_file_source_without_path_raises_configuration_error():
    settings = Settings(_env_file=None, camera_source_type="file", camera_video_file_path=None)
    with pytest.raises(ConfigurationError):
        create_camera(settings)
