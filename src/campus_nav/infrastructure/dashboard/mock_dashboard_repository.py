"""
Mock dashboard repository.

The safe, network-free default (`Settings.dashboard_backend = "mock"`),
following the exact same pattern as `MockObjectDetector`,
`MockGPSProvider`, and every other external-dependency mock in this
codebase: the application boots and every dashboard-dependent code path is
testable with zero network access. This is also the concrete embodiment
of the Phase 1.4 decision to treat the dashboard as replaceable, not
core — swapping to `IITHDashboardRepository` is a one-line settings change,
never a code change to any caller.
"""

from __future__ import annotations

from campus_nav.interfaces.dashboard_interface import (
    BusSchedule,
    CityBusSchedule,
    DashboardRepositoryInterface,
    MessMenu,
)


class MockDashboardRepository(DashboardRepositoryInterface):
    """Returns fixed, caller-configurable data (or nothing, if
    `available=False`), with no network access."""

    def __init__(
        self,
        bus_schedule: BusSchedule | None = None,
        city_bus_schedule: CityBusSchedule | None = None,
        mess_menu: MessMenu | None = None,
        available: bool = True,
    ) -> None:
        self._bus_schedule = bus_schedule
        self._city_bus_schedule = city_bus_schedule
        self._mess_menu = mess_menu
        self._available = available

    def set_available(self, available: bool) -> None:
        """Test/dev helper to simulate the dashboard going offline."""
        self._available = available

    def get_bus_schedule(self) -> BusSchedule | None:
        return self._bus_schedule if self._available else None

    def get_city_bus_schedule(self) -> CityBusSchedule | None:
        return self._city_bus_schedule if self._available else None

    def get_mess_menu(self) -> MessMenu | None:
        return self._mess_menu if self._available else None

    def is_available(self) -> bool:
        return self._available
