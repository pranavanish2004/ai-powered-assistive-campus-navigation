"""
Decision Engine context and response value objects.

Design rationale
-----------------
`AssistantContext` is a deliberately plain data container: it holds
snapshots the API layer has already gathered from the existing Phase 1.2
(Navigation Engine), Phase 1.3 (Vision Service), and Phase 1.4b (Dashboard)
subsystems. The Decision Engine does not reach into those subsystems
itself -- it receives their current state as already-fetched, already-
typed data. This keeps `assistant/` a consumer of `navigation/`, `vision/`,
and the dashboard interface, never the other way around, and keeps every
fusion decision inspectable in a unit test without needing a live
NavigationEngine or VisionService running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from campus_nav.interfaces.dashboard_interface import BusSchedule, MessMenu
from campus_nav.interfaces.mllm_interface import MLLMMessage
from campus_nav.interfaces.obstacle_interface import ObstacleEvent


@dataclass(frozen=True, slots=True)
class NavigationContextSnapshot:
    """A point-in-time snapshot of an active navigation session, if any."""

    state: str
    """NavState value as a string, e.g. "navigating" -- kept as a string
    (not the NavState enum) so this module has no import dependency on
    navigation.engine, matching the same abstraction-boundary pattern used
    for ObstacleEvent in interfaces/obstacle_interface.py."""
    origin_node_id: str | None = None
    destination_node_id: str | None = None
    current_node_id: str | None = None
    distance_remaining_meters: float | None = None
    next_instruction_text: str | None = None


@dataclass(frozen=True, slots=True)
class VisionContextSnapshot:
    """A point-in-time snapshot of what the Vision Pipeline currently
    sees, if the vision service is running."""

    obstacles: list[ObstacleEvent] = field(default_factory=list)
    ocr_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DashboardContextSnapshot:
    """A point-in-time snapshot of dashboard data, fetched only when the
    query appears dashboard-relevant (see prompt_builder.is_dashboard_relevant)."""

    bus_schedule: BusSchedule | None = None
    mess_menu: MessMenu | None = None


@dataclass(frozen=True, slots=True)
class AssistantContext:
    """Everything the Decision Engine has available for one query."""

    navigation: NavigationContextSnapshot | None = None
    vision: VisionContextSnapshot | None = None
    dashboard: DashboardContextSnapshot | None = None
    conversation_history: list[MLLMMessage] = field(default_factory=list)
    """Prior turns in this conversation, oldest first. The Decision Engine
    appends the new user query and the retrieved context to this, but does
    not itself persist history across calls -- the caller (API layer) owns
    conversation lifetime, matching how NavigationEngine instances are
    owned by the API's session registry rather than by NavigationEngine
    itself."""


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """The Decision Engine's final, ready-to-speak answer."""

    text: str
    used_rag: bool
    used_dashboard: bool
    used_navigation_context: bool
    used_vision_context: bool
    source_document_ids: list[str] = field(default_factory=list)
    """Which KnowledgeDocument ids (if any) informed this answer -- lets a
    caller (or a future UI) show "this came from campus knowledge" rather
    than presenting the response as pure LLM invention."""
