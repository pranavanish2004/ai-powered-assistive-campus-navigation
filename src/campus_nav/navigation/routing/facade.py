"""
Routing facade — the single entry point the Navigation Engine should call,
rather than importing `dijkstra_route`/`astar_route` directly.

This is where "accessibility-aware routing" as a *feature* lives: callers
express intent ("give me an accessible route") without needing to know
which underlying algorithm implements it.
"""

from __future__ import annotations

from enum import Enum

from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.routing.astar import astar_route
from campus_nav.navigation.routing.dijkstra import dijkstra_route
from campus_nav.navigation.routing.types import AccessibilityMode, Route


class RoutingAlgorithm(str, Enum):
    DIJKSTRA = "dijkstra"
    ASTAR = "astar"


def find_route(
    graph: CampusGraph,
    start_node_id: str,
    goal_node_id: str,
    algorithm: RoutingAlgorithm = RoutingAlgorithm.ASTAR,
    accessibility_mode: AccessibilityMode = AccessibilityMode.NONE,
) -> Route:
    """Compute a single best route using the requested algorithm.

    A* is the default: it produces the same optimal result as Dijkstra on
    this graph (see astar.py docstring on heuristic admissibility) while
    typically expanding fewer nodes. Dijkstra remains available directly
    for cases where its simpler, heuristic-free behavior is preferred
    (e.g. as a correctness oracle in tests).
    """
    if algorithm is RoutingAlgorithm.DIJKSTRA:
        return dijkstra_route(graph, start_node_id, goal_node_id, accessibility_mode)
    return astar_route(graph, start_node_id, goal_node_id, accessibility_mode)


def find_accessible_route(
    graph: CampusGraph,
    start_node_id: str,
    goal_node_id: str,
    algorithm: RoutingAlgorithm = RoutingAlgorithm.ASTAR,
    strict: bool = True,
) -> Route:
    """Convenience wrapper expressing intent explicitly: compute a route
    that avoids non-accessible infrastructure (stairs, etc.).

    `strict=True` (default) excludes inaccessible edges entirely and raises
    NoRouteFoundError if no fully-accessible path exists — appropriate when
    the user has stated an accessibility requirement and a route containing
    stairs would be actively unsafe to present as "the route", not merely
    suboptimal.

    `strict=False` uses PREFERRED mode: accessible paths are strongly
    favored but an inaccessible edge may appear as a last resort. The
    Navigation Engine should surface `route.uses_inaccessible_edge` to the
    user (via the safety-priority voice lane) if this happens, rather than
    silently routing them over stairs.
    """
    mode = AccessibilityMode.STRICT if strict else AccessibilityMode.PREFERRED
    return find_route(graph, start_node_id, goal_node_id, algorithm, mode)
