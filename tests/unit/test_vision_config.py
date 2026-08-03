"""Tests for Phase 1.3 vision-related Settings fields -- validation and
safe (mock/heuristic) defaults that require no optional [vision] extra."""

import pytest
from pydantic import ValidationError

from campus_nav.config.settings import Settings

pytestmark = pytest.mark.unit


def test_vision_defaults_require_no_optional_extra():
    settings = Settings(_env_file=None)
    assert settings.camera_source_type == "usb"
    assert settings.detector_backend == "mock"
    assert settings.ocr_backend == "mock"
    assert settings.depth_backend == "heuristic"


def test_frame_buffer_size_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, frame_buffer_size=0)


def test_vision_processing_fps_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vision_processing_fps=0)


def test_invalid_camera_source_type_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, camera_source_type="bluetooth")


def test_invalid_detector_backend_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, detector_backend="not-a-real-backend")


def test_invalid_ocr_backend_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ocr_backend="not-a-real-backend")


def test_invalid_depth_backend_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, depth_backend="not-a-real-backend")


def test_ocr_confidence_threshold_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ocr_confidence_threshold=1.5)


def test_depth_safe_distance_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, depth_safe_distance_meters=0)


def test_tracking_iou_threshold_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tracking_iou_threshold=1.5)


def test_tracking_max_missed_frames_negative_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tracking_max_missed_frames=-1)


def test_ip_and_file_camera_settings_default_to_none():
    settings = Settings(_env_file=None)
    assert settings.camera_ip_url is None
    assert settings.camera_video_file_path is None
