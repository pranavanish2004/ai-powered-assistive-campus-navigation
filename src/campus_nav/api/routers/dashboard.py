"""
Dashboard endpoints (Phase 1.4b).

Every endpoint returns HTTP 200 even when the underlying data is
unavailable -- with `available: false` in the body -- rather than a 404 or
500. Per the Phase 1.4 design decision, "the dashboard is unavailable" is
expected, routine state for this replaceable, no-SLA data source, not an
application error.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from campus_nav.api.dependencies import get_dashboard_repository
from campus_nav.api.schemas.dashboard import (
    BusScheduleResponse,
    CityBusScheduleResponse,
    DashboardStatusResponse,
    MessMenuResponse,
)
from campus_nav.config.settings import get_settings
from campus_nav.interfaces.dashboard_interface import DashboardRepositoryInterface

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/status", response_model=DashboardStatusResponse)
def dashboard_status(
    repository: DashboardRepositoryInterface = Depends(get_dashboard_repository),
) -> DashboardStatusResponse:
    settings = get_settings()
    return DashboardStatusResponse(
        is_available=repository.is_available(), backend=settings.dashboard_backend
    )


@router.get("/bus-schedule", response_model=BusScheduleResponse)
def bus_schedule(
    repository: DashboardRepositoryInterface = Depends(get_dashboard_repository),
) -> BusScheduleResponse:
    return BusScheduleResponse.from_domain(repository.get_bus_schedule())


@router.get("/city-bus-schedule", response_model=CityBusScheduleResponse)
def city_bus_schedule(
    repository: DashboardRepositoryInterface = Depends(get_dashboard_repository),
) -> CityBusScheduleResponse:
    return CityBusScheduleResponse.from_domain(repository.get_city_bus_schedule())


@router.get("/mess-menu", response_model=MessMenuResponse)
def mess_menu(
    repository: DashboardRepositoryInterface = Depends(get_dashboard_repository),
) -> MessMenuResponse:
    return MessMenuResponse.from_domain(repository.get_mess_menu())
