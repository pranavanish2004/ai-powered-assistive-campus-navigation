"""Tests for campus_nav.config.settings."""

from campus_nav.config.settings import Settings, get_settings


def test_settings_load_with_defaults():
    """Settings should construct successfully with no environment overrides,
    proving there are safe defaults for every field (important for a fresh
    clone of the repo with no .env file yet)."""
    settings = Settings(_env_file=None)  # ignore any real .env during test
    assert settings.app_env == "development"
    assert settings.camera_index == 0
    assert 0.0 <= settings.min_detection_confidence <= 1.0


def test_get_settings_is_memoized():
    """get_settings() should return the same object on repeated calls
    (lru_cache), so we parse configuration exactly once per process."""
    assert get_settings() is get_settings()


def test_invalid_app_env_rejected():
    """Settings validation should fail fast on an invalid literal value,
    rather than silently accepting a typo like 'developmnet'."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="not-a-real-environment")
