"""Tests for Phase 1.4b dashboard Settings fields and the backend factory."""

import pytest
from pydantic import ValidationError

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.dashboard.dashboard_factory import create_dashboard_repository
from campus_nav.infrastructure.dashboard.iith_dashboard_repository import IITHDashboardRepository
from campus_nav.infrastructure.dashboard.mock_dashboard_repository import MockDashboardRepository

pytestmark = pytest.mark.unit


def test_dashboard_defaults_require_no_network_access():
    settings = Settings(_env_file=None)
    assert settings.dashboard_backend == "mock"
    assert settings.dashboard_base_url == "https://api.iith.dev"
    assert settings.dashboard_timeout_seconds > 0


def test_invalid_dashboard_backend_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, dashboard_backend="not-a-real-backend")


def test_dashboard_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, dashboard_timeout_seconds=0)


def test_factory_default_returns_mock_repository():
    settings = Settings(_env_file=None)
    repo = create_dashboard_repository(settings)
    assert isinstance(repo, MockDashboardRepository)


def test_factory_iith_live_backend_returns_real_repository():
    settings = Settings(_env_file=None, dashboard_backend="iith_live")
    repo = create_dashboard_repository(settings)
    assert isinstance(repo, IITHDashboardRepository)
    assert repo._base_url == "https://api.iith.dev"


def test_factory_passes_configured_base_url_and_timeout():
    settings = Settings(
        _env_file=None,
        dashboard_backend="iith_live",
        dashboard_base_url="https://staging.example.com",
        dashboard_timeout_seconds=2.0,
    )
    repo = create_dashboard_repository(settings)
    assert isinstance(repo, IITHDashboardRepository)
    assert repo._base_url == "https://staging.example.com"
    assert repo._timeout_seconds == 2.0
