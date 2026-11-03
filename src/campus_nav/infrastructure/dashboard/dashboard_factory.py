"""Dashboard repository factory -- Settings-driven backend selection,
mirroring camera_factory.py and vision_factory.py."""

from __future__ import annotations

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.dashboard.iith_dashboard_repository import IITHDashboardRepository
from campus_nav.infrastructure.dashboard.mock_dashboard_repository import MockDashboardRepository
from campus_nav.interfaces.dashboard_interface import DashboardRepositoryInterface


def create_dashboard_repository(settings: Settings) -> DashboardRepositoryInterface:
    if settings.dashboard_backend == "iith_live":
        return IITHDashboardRepository(
            base_url=settings.dashboard_base_url,
            timeout_seconds=settings.dashboard_timeout_seconds,
        )
    return MockDashboardRepository()
