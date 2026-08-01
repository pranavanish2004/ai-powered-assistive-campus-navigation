"""
A* routing, using great-circle (haversine) distance to the goal as the
heuristic.

The heuristic is admissible (never overestimates) because straight-line
distance is always <= any real path distance across campus paths/corridors,
which guarantees A* finds the same optimal route as Dijkstra while
typically visiting far fewer nodes — the benefit grows with graph size,
which matters once the full IIT-H campus graph (hundreds of nodes) replaces
the small pilot fixture used in tests.
"""

from __future__ import annotations

import heapq
import math

from campus_nav.core.exceptions import NoRouteFoundError
from campus_nav.navigation.graph.geometry import haversine_distance_meters
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.routing.types import AccessibilityMode, Route, edge_traversal_weight


def astar_route(
    graph: CampusGraph,
    start_node_id: str,
    goal_node_id: str,
    accessibility_mode: AccessibilityMode = AccessibilityMode.NONE,
) -> Route:
    """Compute the shortest route from `start_node_id` to `goal_node_id`
    using A* search. Same contract/exceptions as `dijkstra_route`."""
    graph.get_node(start_node_id)
    goal_node = graph.get_node(goal_node_id)

    if start_node_id == goal_node_id:
        return Route.from_path(graph, [start_node_id], [], "astar", accessibility_mode)

    def heuristic(node_id: str) -> float:
        node = graph.get_node(node_id)
        return haversine_distance_meters(
            node.latitude, node.longitude, goal_node.latitude, goal_node.longitude
        )

    g_score: dict[str, float] = {start_node_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}
    visited: set[str] = set()

    counter = 0  # tie-breaker so heap comparisons never fall back to comparing node ids
    open_set: list[tuple[float, int, str]] = [(heuristic(start_node_id), counter, start_node_id)]

    while open_set:
        _, _, current_node = heapq.heappop(open_set)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal_node_id:
            break

        for edge, neighbor in graph.neighbors(current_node):
            weight = edge_traversal_weight(edge, accessibility_mode)
            if math.isinf(weight):
                continue

            tentative_g = g_score[current_node] + weight
            if tentative_g < g_score.get(neighbor.id, math.inf):
                g_score[neighbor.id] = tentative_g
                previous[neighbor.id] = (current_node, edge.id)
                counter += 1
                f_score = tentative_g + heuristic(neighbor.id)
                heapq.heappush(open_set, (f_score, counter, neighbor.id))

    if goal_node_id not in g_score:
        raise NoRouteFoundError(
            f"No route from {start_node_id!r} to {goal_node_id!r} "
            f"under accessibility mode {accessibility_mode.value!r}."
        )

    node_ids, edge_ids = _reconstruct_path(previous, start_node_id, goal_node_id)
    return Route.from_path(graph, node_ids, edge_ids, "astar", accessibility_mode)


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
