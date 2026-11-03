"""
IITH Dashboard repository -- the real HTTP-backed implementation.

Only calls the three endpoints confirmed, by reading the actual backend
source, to require no authentication:

    GET /transport/          -> campus bus + EV schedule
    GET /transport/cityBus   -> city bus schedule
    GET /mess_menu/          -> current week's mess menu

Every other feature of that application (timetable, cab-sharing, lost &
found, marketplace) requires a real @iith.ac.in Google-authenticated
session and is intentionally NOT implemented here -- this assistant has
no mechanism to obtain or hold a specific student's credentials, and
should not pretend to.

Failure handling
-----------------
This is exactly the kind of external dependency the Phase 1.4 design
decision was about: the dashboard is a community-maintained, no-SLA,
no-rate-limiting side project, not institute infrastructure. Every method
catches network/timeout/parse failures and returns None (via
`is_available()` semantics) rather than raising -- a slow or down
dashboard must never block or degrade navigation, vision, or RAG.
"""

from __future__ import annotations

from typing import Any

import httpx

from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.dashboard_interface import (
    BusSchedule,
    CityBusSchedule,
    DashboardRepositoryInterface,
    MessMenu,
)

logger = get_logger(__name__)


class IITHDashboardRepository(DashboardRepositoryInterface):
    """HTTP client for the public endpoints of the IITH Dashboard
    (`https://api.iith.dev` in production; overridable for self-hosted
    forks or testing)."""

    def __init__(self, base_url: str, timeout_seconds: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._last_check_available = True
        """Cheap, best-effort liveness flag updated by whichever call ran
        most recently -- not a separate health-check request, since the
        Dashboard has no dedicated health endpoint and this project should
        not add load to a no-SLA community service just to poll it."""

    def _get(self, path: str) -> Any | None:
        url = f"{self._base_url}{path}"
        try:
            response = httpx.get(url, timeout=self._timeout_seconds)
            response.raise_for_status()
            self._last_check_available = True
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError covers response.json() failing on non-JSON bodies.
            logger.info(
                "IITH Dashboard request failed; treating as unavailable",
                extra={"url": url, "error": str(exc)},
            )
            self._last_check_available = False
            return None

    def get_bus_schedule(self) -> BusSchedule | None:
        data = self._get("/transport/")
        if data is None:
            return None
        try:
            return BusSchedule(
                routes=dict(data.get("bus", {})),
                ev_routes=dict(data.get("EV", {})),
            )
        except AttributeError:
            logger.warning("Unexpected /transport/ response shape", extra={"data": str(data)[:200]})
            return None

    def get_city_bus_schedule(self) -> CityBusSchedule | None:
        data = self._get("/transport/cityBus")
        if data is None:
            return None
        return CityBusSchedule(raw=data if isinstance(data, dict) else {"data": data})

    def get_mess_menu(self) -> MessMenu | None:
        data = self._get("/mess_menu/")
        if data is None:
            return None
        try:
            return MessMenu(days=dict(data))
        except (TypeError, ValueError):
            logger.warning("Unexpected /mess_menu/ response shape", extra={"data": str(data)[:200]})
            return None

    def is_available(self) -> bool:
        return self._last_check_available
