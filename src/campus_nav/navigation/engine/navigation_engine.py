"""
NavigationEngine: the Phase 1.2 orchestrator.

Wires together, per active navigation session:
  routing (facade.find_route) -> the active Route
  localization (localize_fix) -> where the user is now, given a raw fix
  SafetyEngine -> deterministic route-adherence / confidence decisions
  NavigationStateMachine -> the authoritative safety state
  instructions (generate_instructions) -> what to say next
  VoiceGuidance -> actually saying it, with correct SpeechPriority

`update_position()` receives the raw (latitude, longitude, accuracy) fix as
explicit arguments -- this matches how a real deployment will actually
supply position (a mobile client's GPS reading arriving over the API, or a
test harness) rather than requiring the engine to poll an injected
GPSInterface itself. The GPSInterface + GraphSnappingLocalizer combination
in navigation/localization/localizer.py remains available for an alternate
usage pattern (e.g. a local daemon polling hardware GPS directly), but is
not used here -- see that module's docstring.

This class deliberately contains no routing math, no snapping math, and no
TTS engine code of its own -- it only sequences calls to the modules that
do, which is what keeps each of those modules independently unit-testable.
"""

from __future__ import annotations

from campus_nav.core.exceptions import NavigationError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.localization_interface import LocalizationResult
from campus_nav.interfaces.navigation_interface import (
    NavEvent,
    NavigationEngineInterface,
    NavigationInstruction,
    NavigationUpdate,
    NavState,
)
from campus_nav.navigation.engine.instructions import generate_instructions
from campus_nav.navigation.engine.safety_engine import SafetyEngine
from campus_nav.navigation.engine.state_machine import NavigationStateMachine
from campus_nav.navigation.engine.voice_guidance import VoiceGuidance
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.localization.localizer import localize_fix
from campus_nav.navigation.routing.facade import RoutingAlgorithm, find_route
from campus_nav.navigation.routing.types import AccessibilityMode, Route

logger = get_logger(__name__)


class NavigationEngine(NavigationEngineInterface):
    """Coordinates one active navigation session for one user."""

    def __init__(
        self,
        graph: CampusGraph,
        voice_guidance: VoiceGuidance,
        max_snap_distance_meters: float = 30.0,
        off_route_warning_meters: float = 15.0,
        routing_algorithm: RoutingAlgorithm = RoutingAlgorithm.ASTAR,
    ) -> None:
        self._graph = graph
        self._voice = voice_guidance
        self._max_snap_distance_meters = max_snap_distance_meters
        self._routing_algorithm = routing_algorithm
        self._safety_engine = SafetyEngine(off_route_warning_meters=off_route_warning_meters)
        self._state_machine = NavigationStateMachine()

        self._route: Route | None = None
        self._instructions: list[NavigationInstruction] = []
        self._next_instruction_index: int = 0
        self._destination_node_id: str | None = None
        self._last_announced_instruction: NavigationInstruction | None = None
        self._last_known_node_id: str | None = None

    @property
    def state(self) -> NavState:
        return self._state_machine.state

    @property
    def active_route(self) -> Route | None:
        return self._route

    @property
    def last_instruction(self) -> NavigationInstruction | None:
        """The most recently announced NavigationInstruction, if any. Lets
        callers (e.g. the API layer) surface the current instruction text
        without reaching into VoiceGuidance/TTS internals."""
        return self._last_announced_instruction

    @property
    def last_known_node_id(self) -> str | None:
        """The most recently snapped node id, if any position update has
        succeeded since navigation started."""
        return self._last_known_node_id

    def start_navigation(
        self,
        origin_node_id: str,
        destination_node_id: str,
        accessibility_mode: AccessibilityMode = AccessibilityMode.NONE,
    ) -> Route:
        route = find_route(
            self._graph,
            origin_node_id,
            destination_node_id,
            algorithm=self._routing_algorithm,
            accessibility_mode=accessibility_mode,
        )

        self._route = route
        self._destination_node_id = destination_node_id
        self._instructions = generate_instructions(route, self._graph)
        self._next_instruction_index = 0

        self._state_machine.transition(NavEvent.START)
        logger.info(
            "Navigation started",
            extra={
                "origin": origin_node_id,
                "destination": destination_node_id,
                "distance_meters": route.total_distance_meters,
                "algorithm": route.algorithm,
            },
        )

        if self._instructions:
            self._voice.announce_instruction(self._instructions[0])
            self._last_announced_instruction = self._instructions[0]
            self._next_instruction_index = 1

        return route

    def update_position(
        self, latitude: float, longitude: float, gps_accuracy_meters: float
    ) -> NavigationUpdate:
        if self._route is None:
            raise NavigationError("update_position() called with no active route.")

        # Position updates are only meaningful while actively navigating,
        # uncertain (trying to recover), or rerouting.
        if self.state not in (NavState.NAVIGATING, NavState.UNCERTAIN, NavState.REROUTING):
            return NavigationUpdate(
                state=self.state,
                current_node_id=None,
                next_instruction=None,
                safety_alert=None,
                distance_remaining_meters=None,
            )

        localization: LocalizationResult | None = localize_fix(
            self._graph, latitude, longitude, gps_accuracy_meters, self._max_snap_distance_meters
        )
        decision = self._safety_engine.evaluate(localization, self._route, self._graph)

        safety_alert = decision.alert_text
        if safety_alert:
            self._voice.announce_safety_alert(safety_alert)

        if decision.event is not None and self._state_machine.can_transition(decision.event):
            self._state_machine.transition(decision.event)

            if decision.event is NavEvent.REROUTE_NEEDED and localization is not None:
                self._reroute_from(localization.nearest_node.id)

        next_instruction = None
        if self.state is NavState.NAVIGATING and self._next_instruction_index < len(
            self._instructions
        ):
            next_instruction = self._instructions[self._next_instruction_index]
            self._voice.announce_instruction(next_instruction)
            self._last_announced_instruction = next_instruction
            self._next_instruction_index += 1

        distance_remaining = self._route.total_distance_meters if self._route else None

        if localization is not None:
            self._last_known_node_id = localization.nearest_node.id

        return NavigationUpdate(
            state=self.state,
            current_node_id=localization.nearest_node.id if localization else None,
            next_instruction=next_instruction,
            safety_alert=safety_alert,
            distance_remaining_meters=distance_remaining,
        )

    def _reroute_from(self, current_node_id: str) -> None:
        """Recompute the route from the user's current snapped position to
        the original destination, then return to NAVIGATING."""
        assert self._destination_node_id is not None
        previous_mode = self._route.accessibility_mode if self._route else AccessibilityMode.NONE
        try:
            new_route = find_route(
                self._graph,
                current_node_id,
                self._destination_node_id,
                algorithm=self._routing_algorithm,
                accessibility_mode=previous_mode,
            )
        except Exception:
            logger.warning("Reroute failed; remaining in REROUTING state", exc_info=True)
            return

        self._route = new_route
        self._instructions = generate_instructions(new_route, self._graph)
        self._next_instruction_index = 0
        self._state_machine.transition(NavEvent.REROUTE_COMPLETE)
        logger.info("Reroute complete", extra={"from_node": current_node_id})

    def pause(self) -> None:
        self._state_machine.transition(NavEvent.PAUSE)

    def resume(self) -> None:
        self._state_machine.transition(NavEvent.RESUME)

    def stop(self) -> None:
        self._state_machine.transition(NavEvent.STOP)
        self._route = None
        self._instructions = []
        self._next_instruction_index = 0
        self._destination_node_id = None
