"""
Dijkstra's algorithm for shortest-path routing over the campus graph.

Uses a binary heap (heapq) priority queue, O((V+E) log V). This is the
"always correct, works for any non-negative weights" baseline routing
algorithm; `astar.py` provides a heuristically-guided faster alternative for
the same graph.
"""

from __future__ import annotations

import heapq
import math

from campus_nav.core.exceptions import NoRouteFoundError
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.routing.types import AccessibilityMode, Route, edge_traversal_weight


def dijkstra_route(
    graph: CampusGraph,
    start_node_id: str,
    goal_node_id: str,
    accessibility_mode: AccessibilityMode = AccessibilityMode.NONE,
) -> Route:
    """Compute the shortest route from `start_node_id` to `goal_node_id`.

    Raises NodeNotFoundError (via graph.get_node, called implicitly through
    graph.neighbors) if either endpoint doesn't exist, and NoRouteFoundError
    if no path exists under the given accessibility constraint.
    """
    # Validate endpoints exist up front for a clear error before any search work.
    graph.get_node(start_node_id)
    graph.get_node(goal_node_id)

    if start_node_id == goal_node_id:
        return Route.from_path(graph, [start_node_id], [], "dijkstra", accessibility_mode)

    distances: dict[str, float] = {start_node_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}  # node_id -> (prev_node_id, edge_id)
    visited: set[str] = set()
    queue: list[tuple[float, str]] = [(0.0, start_node_id)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal_node_id:
            break

        for edge, neighbor in graph.neighbors(current_node):
            weight = edge_traversal_weight(edge, accessibility_mode)
            if math.isinf(weight):
                continue  # STRICT mode: this edge is impassable

            candidate_distance = current_distance + weight
            if candidate_distance < distances.get(neighbor.id, math.inf):
                distances[neighbor.id] = candidate_distance
                previous[neighbor.id] = (current_node, edge.id)
                heapq.heappush(queue, (candidate_distance, neighbor.id))

    if goal_node_id not in distances:
        raise NoRouteFoundError(
            f"No route from {start_node_id!r} to {goal_node_id!r} "
            f"under accessibility mode {accessibility_mode.value!r}."
        )

    node_ids, edge_ids = _reconstruct_path(previous, start_node_id, goal_node_id)
    return Route.from_path(graph, node_ids, edge_ids, "dijkstra", accessibility_mode)


def _reconstruct_path(
    previous: dict[str, tuple[str, str]], start_node_id: str, goal_node_id: str
) -> tuple[list[str], list[str]]:
    node_ids = [goal_node_id]
    edge_ids: list[str] = []
    current = goal_node_id
    while current != start_node_id:
        prev_node, edge_id = previous[current]
        edge_ids.append(edge_id)
        node_ids.append(prev_node)
        current = prev_node
    node_ids.reverse()
    edge_ids.reverse()
    return node_ids, edge_ids
