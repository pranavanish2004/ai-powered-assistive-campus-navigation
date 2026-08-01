"""
Turn-by-turn instruction engine.

Converts a computed Route (a list of node/edge ids) into a sequence of
short, speakable NavigationInstruction objects — "Continue for 40 meters",
"Turn left", "Ramp ahead", "You have arrived at the Library entrance."

This deliberately produces plain, literal instructions with no LLM
involvement — turn-by-turn guidance is exactly the kind of deterministic,
safety-relevant output that belongs in the SAFETY-adjacent conversational
lane described in docs/architecture.md, not something an LLM should
paraphrase or improvise.
"""

from __future__ import annotations

from campus_nav.interfaces.navigation_interface import NavigationInstruction
from campus_nav.navigation.graph.geometry import bearing_difference_degrees, initial_bearing_degrees
from campus_nav.navigation.graph.models import CampusGraph, PathType
from campus_nav.navigation.routing.types import Route

# Turn classification thresholds, in degrees of bearing change.
_STRAIGHT_THRESHOLD = 20.0
_SLIGHT_THRESHOLD = 60.0
_SHARP_THRESHOLD = 120.0

_PATH_TYPE_PHRASES: dict[PathType, str] = {
    PathType.STAIRS: "Stairs ahead.",
    PathType.RAMP: "Ramp ahead.",
    PathType.ROAD_CROSSING: "Road crossing ahead. Listen carefully for traffic.",
}


def _classify_turn(bearing_change: float) -> str:
    magnitude = abs(bearing_change)
    if magnitude <= _STRAIGHT_THRESHOLD:
        return "Continue straight"
    if magnitude <= _SLIGHT_THRESHOLD:
        return "Turn slightly right" if bearing_change > 0 else "Turn slightly left"
    if magnitude <= _SHARP_THRESHOLD:
        return "Turn right" if bearing_change > 0 else "Turn left"
    return "Turn sharply right" if bearing_change > 0 else "Turn sharply left"


def generate_instructions(route: Route, graph: CampusGraph) -> list[NavigationInstruction]:
    """Generate one instruction per edge in the route, plus a final arrival
    instruction. Returns an empty list for a route with fewer than 2 nodes
    (i.e. already at the destination)."""
    if route.is_empty():
        return []

    instructions: list[NavigationInstruction] = []
    previous_bearing: float | None = None

    for index, edge_id in enumerate(route.edge_ids):
        edge = graph.get_edge(edge_id)
        from_node = graph.get_node(route.node_ids[index])
        to_node = graph.get_node(route.node_ids[index + 1])

        bearing = initial_bearing_degrees(
            from_node.latitude, from_node.longitude, to_node.latitude, to_node.longitude
        )

        phrase_parts: list[str] = []
        if previous_bearing is not None:
            turn_phrase = _classify_turn(bearing_difference_degrees(previous_bearing, bearing))
            if turn_phrase != "Continue straight" or index == 0:
                phrase_parts.append(turn_phrase + ".")
        else:
            phrase_parts.append(f"Head toward {to_node.name}.")

        path_hint = _PATH_TYPE_PHRASES.get(edge.path_type)
        if path_hint:
            phrase_parts.append(path_hint)

        phrase_parts.append(f"Continue for {round(edge.distance_meters)} meters.")

        instructions.append(
            NavigationInstruction(
                text=" ".join(phrase_parts),
                target_node_id=to_node.id,
                distance_meters=edge.distance_meters,
                is_arrival=False,
            )
        )
        previous_bearing = bearing

    destination = graph.get_node(route.node_ids[-1])
    instructions.append(
        NavigationInstruction(
            text=f"You have arrived at {destination.name}.",
            target_node_id=destination.id,
            distance_meters=0.0,
            is_arrival=True,
        )
    )
    return instructions
