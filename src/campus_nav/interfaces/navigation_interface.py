"""
Navigation engine abstraction and shared navigation value types.

`NavState`/`NavEvent` live here (not in navigation/engine/) so that
higher-level consumers (the API layer, tests, the future Decision Engine)
can depend on these stable, abstract types without importing the concrete
NavigationEngine implementation — the concrete state machine in
navigation/engine/state_machine.py depends on these types, not the other
way around, keeping the dependency arrow pointing from concrete -> abstract
per Clean Architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from campus_nav.navigation.routing.types import AccessibilityMode, Route


class NavState(str, Enum):
    """The Navigation Engine's safety-relevant state at any moment. This is
    intentionally small and exhaustive — every navigation-affecting
    decision (should the Safety Engine speak? should the API report
    "arrived"?) should be answerable by inspecting this single value."""

    IDLE = "idle"
    NAVIGATING = "navigating"
    PAUSED = "paused"
    UNCERTAIN = "uncertain"
    REROUTING = "rerouting"
    ARRIVED = "arrived"


class NavEvent(str, Enum):
    """An event that may trigger a state transition. See
    navigation/engine/state_machine.py for the allowed transition table."""

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    ARRIVE = "arrive"
    LOSE_CONFIDENCE = "lose_confidence"
    REGAIN_CONFIDENCE = "regain_confidence"
    REROUTE_NEEDED = "reroute_needed"
    REROUTE_COMPLETE = "reroute_complete"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class NavigationInstruction:
    """A single turn-by-turn guidance message."""

    text: str
    target_node_id: str | None
    distance_meters: float
    is_arrival: bool = False


@dataclass(frozen=True, slots=True)
class NavigationUpdate:
    """The result of feeding a new position fix into the Navigation Engine
    — everything the caller (API layer, voice guidance) needs to react."""

    state: NavState
    current_node_id: str | None
    next_instruction: NavigationInstruction | None
    safety_alert: str | None
    """Non-None only when the deterministic Safety Engine decided the user
    must be told something immediately (off-route, low confidence, etc.).
    Always delivered via SpeechPriority.SAFETY when spoken."""
    distance_remaining_meters: float | None


class NavigationEngineInterface(ABC):
    """Abstract base class for the Navigation Engine. The API layer and any
    future Decision Engine should depend on this, not the concrete
    NavigationEngine, so alternate orchestration strategies (e.g. a
    multi-destination trip planner) can be substituted in tests."""

    @abstractmethod
    def start_navigation(
        self,
        origin_node_id: str,
        destination_node_id: str,
        accessibility_mode: AccessibilityMode = AccessibilityMode.NONE,
    ) -> Route:
        """Compute an initial route and transition to NAVIGATING."""

    @abstractmethod
    def update_position(
        self, latitude: float, longitude: float, gps_accuracy_meters: float
    ) -> NavigationUpdate:
        """Feed a new raw position fix into the engine and receive the
        resulting navigation state/instruction/alert."""

    @abstractmethod
    def pause(self) -> None:
        """Transition to PAUSED. Safe to call only while NAVIGATING."""

    @abstractmethod
    def resume(self) -> None:
        """Transition back to NAVIGATING. Safe to call only while PAUSED."""

    @abstractmethod
    def stop(self) -> None:
        """End the navigation session, transitioning to IDLE from any
        active state."""

    @property
    @abstractmethod
    def state(self) -> NavState:
        """The engine's current NavState."""
