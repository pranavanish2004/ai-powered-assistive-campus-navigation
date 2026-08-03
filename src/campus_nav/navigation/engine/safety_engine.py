"""
Deterministic Safety Engine.

Design rationale
-----------------
Per the Phase 1.1 architecture review: safety-relevant decisions must never
depend on an LLM call, both for latency (an LLM might take seconds) and for
predictability (a safety rule must behave the same way every time, not be
subject to generative variance). This engine is pure, deterministic Python:
given a localization result and the active route, it decides whether the
user has drifted off-route, whether confidence has dropped too low to
trust, or whether they've arrived — and produces the exact alert text to
speak, with no LLM in the loop.

Vision-based obstacle detection (Phase 1.3) will feed into a *different*
part of this same engine's mandate — this class's Phase 1.2 scope is
route-adherence and localization-confidence monitoring only, since no
camera/YOLO pipeline exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from campus_nav.interfaces.localization_interface import LocalizationResult
from campus_nav.interfaces.navigation_interface import NavEvent
from campus_nav.interfaces.obstacle_interface import ObstacleEvent
from campus_nav.navigation.graph.geometry import haversine_distance_meters
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.routing.types import Route

_LOW_CONFIDENCE_THRESHOLD = 0.3
_ARRIVAL_RADIUS_METERS = 5.0


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """The Safety Engine's verdict for one position update. `event`, if
    present, should be fed into the NavigationStateMachine. `alert_text`, if
    present, must always be spoken via SpeechPriority.SAFETY."""

    event: NavEvent | None
    alert_text: str | None


class SafetyEngine:
    """Rule-based, deterministic safety monitor for an active navigation
    session."""

    def __init__(
        self,
        off_route_warning_meters: float,
        low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
        arrival_radius_meters: float = _ARRIVAL_RADIUS_METERS,
    ) -> None:
        self._off_route_warning_meters = off_route_warning_meters
        self._low_confidence_threshold = low_confidence_threshold
        self._arrival_radius_meters = arrival_radius_meters

    def evaluate(
        self,
        localization: LocalizationResult | None,
        route: Route,
        graph: CampusGraph,
    ) -> SafetyDecision:
        """Evaluate the current position against the active route.

        Priority order matters: a total loss of position (None) is checked
        first (nothing else can be evaluated without a position), then low
        confidence, then arrival, then off-route drift. Only one decision is
        returned per call — the Navigation Engine calls this on every
        position update, so a lower-priority issue will simply be
        re-evaluated (and potentially reported) on the next update.
        """
        if localization is None:
            return SafetyDecision(
                event=NavEvent.LOSE_CONFIDENCE,
                alert_text=(
                    "Losing position signal. Please stop and wait, "
                    "or continue carefully using your own judgment."
                ),
            )

        if localization.confidence < self._low_confidence_threshold:
            return SafetyDecision(
                event=NavEvent.LOSE_CONFIDENCE,
                alert_text="Position confidence is low. Proceed carefully.",
            )

        destination_node = graph.get_node(route.node_ids[-1])
        distance_to_destination = haversine_distance_meters(
            localization.latitude,
            localization.longitude,
            destination_node.latitude,
            destination_node.longitude,
        )
        if distance_to_destination <= self._arrival_radius_meters:
            return SafetyDecision(event=NavEvent.ARRIVE, alert_text=None)

        distance_to_route = self._distance_to_route(localization, route, graph)
        if distance_to_route > self._off_route_warning_meters:
            return SafetyDecision(
                event=NavEvent.REROUTE_NEEDED,
                alert_text="You have moved off the route. Recalculating.",
            )

        return SafetyDecision(event=None, alert_text=None)

    def evaluate_obstacles(self, obstacle_events: list[ObstacleEvent]) -> SafetyDecision:
        """Evaluate vision-derived obstacle observations for this frame
        (Phase 1.3 integration point).

        Deliberately separate from `evaluate()` rather than merged into
        it: route-adherence/confidence and obstacle-avoidance are
        independent safety concerns with independent inputs (a
        `LocalizationResult` + `Route` vs. a list of `ObstacleEvent`s from
        the Vision Pipeline), and keeping them as separate methods means
        neither Phase 1.2's route logic nor this method's obstacle logic
        needs to know about the other's data shapes. Both still funnel
        through the same `SafetyDecision` type and the same SAFETY-priority
        voice lane (see `VoiceGuidance.announce_safety_alert`) -- there is
        exactly one way obstacle warnings and route warnings reach the
        user, they just originate from different evaluation paths.

        Returns the first blocking obstacle's announcement (if any) with
        `event=None` -- obstacle events do not themselves drive
        `NavigationStateMachine` transitions in Phase 1.3 (that fusion,
        e.g. "reroute around a persistent obstacle", is explicitly a
        Phase 1.4 Decision Engine responsibility) — this method only
        decides whether something must be said *right now*.
        """
        for event in obstacle_events:
            if event.announce_text:
                return SafetyDecision(event=None, alert_text=event.announce_text)
        return SafetyDecision(event=None, alert_text=None)

    @staticmethod
    def _distance_to_route(
        localization: LocalizationResult, route: Route, graph: CampusGraph
    ) -> float:
        """Approximate distance from the current position to the route, as
        the minimum distance to any node on the route. This is a
        deliberately cheap approximation (exact would project onto every
        route edge segment) that is more than adequate for corridor-scale
        campus routes, where nodes are placed every few tens of meters."""
        return min(
            haversine_distance_meters(
                localization.latitude,
                localization.longitude,
                graph.get_node(node_id).latitude,
                graph.get_node(node_id).longitude,
            )
            for node_id in route.node_ids
        )
