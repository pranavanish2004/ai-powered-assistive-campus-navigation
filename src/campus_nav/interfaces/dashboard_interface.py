"""
Campus dashboard abstraction.

Design rationale
-----------------
This interface is deliberately narrow. Phase 1.4 research (reading the
actual open-source IITH Dashboard backend, not its marketing page) found
that only three endpoints are genuinely public and unauthenticated:
bus schedule, city bus schedule, and the current week's mess menu.
Everything else in that application (timetable, cab-sharing, lost & found,
marketplace) requires a real `@iith.ac.in` Google-authenticated session,
which our assistant cannot transparently obtain on a user's behalf.

More importantly: that backend has **no campus wayfinding data at all** —
no buildings, departments, library, medical centre, faculty offices, or
accessibility information. It is a student-life utility app, not a
campus directory. This confirms, concretely, the architectural decision
already made for this project: the Dashboard is a *replaceable enhancement*
(bus times, mess menu) layered on top of a system that must otherwise
function entirely on the Campus Graph, GPS, Vision, OCR, and the RAG/local
knowledge base. `DashboardRepositoryInterface` exists only to answer "what
enhancement data is available right now," never "where is X" — that
question belongs to RAG and the campus graph, not this interface, and
always will, even if a richer dashboard becomes available later.

Every method returns `None` on unavailability (matching the pattern
established by `GPSInterface.get_current_location()` and
`LocalizationInterface.get_current_position()`) rather than raising —
"the dashboard has nothing to say right now" is expected, routine control
flow for a data source explicitly designed to be optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BusSchedule:
    """Campus shuttle and EV schedule, keyed by route/direction name to a
    list of departure times as strings (e.g. "08:15:00") — kept as strings
    rather than parsed `time` objects because the upstream source's own
    format is not guaranteed stable, and re-parsing here would just move
    the failure point rather than remove it."""

    routes: dict[str, list[str]] = field(default_factory=dict)
    ev_routes: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CityBusSchedule:
    """City bus schedule. Deliberately generic (`raw` dict) rather than a
    strongly-typed shape: unlike the campus bus schedule, no scraper
    script confirming this endpoint's exact structure was found during
    research — representing it precisely here would be documenting a
    guess as if it were a verified fact."""

    raw: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MessMenu:
    """One week's mess menu: day name -> meal name -> list of food items,
    e.g. `menu.days["Monday"]["Breakfast"] == ["Idli", "Sambar", ...]`.
    Confirmed against the actual menu-generation script's output shape."""

    days: dict[str, dict[str, list[str]]] = field(default_factory=dict)


class DashboardRepositoryInterface(ABC):
    """Abstract base class for campus dashboard data sources.

    Deliberately small: exactly the data confirmed to exist and be
    reachable without authentication (see module docstring). Do not add
    methods here speculatively for data this research did not confirm
    exists (building directories, faculty info, events) — that belongs to
    the RAG/local knowledge base, not this interface.
    """

    @abstractmethod
    def get_bus_schedule(self) -> BusSchedule | None:
        """Return the current campus bus/EV schedule, or None if the
        dashboard is unavailable."""

    @abstractmethod
    def get_city_bus_schedule(self) -> CityBusSchedule | None:
        """Return the current city bus schedule, or None if unavailable."""

    @abstractmethod
    def get_mess_menu(self) -> MessMenu | None:
        """Return the current week's mess menu, or None if unavailable."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the dashboard data source is currently reachable.
        Callers (e.g. the future Decision Engine) should use this to decide
        whether to even attempt a query, rather than treating a None
        return from the methods above as a surprise."""
