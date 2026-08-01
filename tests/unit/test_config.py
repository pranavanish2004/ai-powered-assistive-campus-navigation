"""Tests for campus_nav.config.settings."""

import pytest
from pydantic import ValidationError

from campus_nav.config.settings import Settings, get_settings

pytestmark = pytest.mark.unit


def test_settings_load_with_defaults():
    settings = Settings(_env_file=None)
    assert settings.app_env == "development"
    assert settings.camera_index == 0
    assert 0.0 <= settings.min_detection_confidence <= 1.0
    assert settings.gps_snap_max_distance_meters > 0
    assert settings.off_route_warning_meters > 0


def test_get_settings_is_memoized():
    assert get_settings() is get_settings()


def test_invalid_app_env_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="not-a-real-environment")


def test_invalid_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, min_detection_confidence=1.5)


def test_invalid_log_format_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_format="xml")


def test_non_positive_camera_dimensions_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, camera_frame_width=0)
