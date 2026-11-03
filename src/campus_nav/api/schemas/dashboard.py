"""Pydantic response schemas for the /dashboard endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from campus_nav.interfaces.dashboard_interface import BusSchedule, CityBusSchedule, MessMenu


class BusScheduleResponse(BaseModel):
    available: bool
    routes: dict[str, list[str]] = Field(default_factory=dict)
    ev_routes: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, schedule: BusSchedule | None) -> BusScheduleResponse:
        if schedule is None:
            return cls(available=False)
        return cls(available=True, routes=schedule.routes, ev_routes=schedule.ev_routes)


class CityBusScheduleResponse(BaseModel):
    available: bool
    raw: dict = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, schedule: CityBusSchedule | None) -> CityBusScheduleResponse:
        if schedule is None:
            return cls(available=False)
        return cls(available=True, raw=schedule.raw)


class MessMenuResponse(BaseModel):
    available: bool
    days: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, menu: MessMenu | None) -> MessMenuResponse:
        if menu is None:
            return cls(available=False)
        return cls(available=True, days=menu.days)


class DashboardStatusResponse(BaseModel):
    is_available: bool
    backend: str
