"""
Multiple route generation (k-shortest paths) via Yen's algorithm.

Why this exists
-----------------
A visually impaired user (or the Navigation Engine on their behalf) may want
an alternative to the single shortest route — e.g. "give me a second option
in case the first one has construction blocking it," or so the Decision
Engine can offer a route with fewer road crossings even if it's slightly
longer. Yen's algorithm builds on top of plain Dijkstra (no separate
all-paths search needed) by systematically excluding sub-paths of
already-found routes and re-running shortest-path search on the remainder.
"""

from __future__ import annotations

import heapq
import math

from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.routing.types import AccessibilityMode, Route, edge_traversal_weight


def k_shortest_routes(
    graph: CampusGraph,
    start_node_id: str,
    goal_node_id: str,
    k: int = 3,
    accessibility_mode: AccessibilityMode = AccessibilityMode.NONE,
) -> list[Route]:
    """Return up to `k` distinct routes from start to goal, ordered from
    shortest to longest. Returns fewer than `k` routes (possibly zero) if
    that many distinct paths don't exist — this is not an error, since
    "only one route exists" is a completely normal campus-graph situation.
    """
    graph.get_node(start_node_id)
    graph.get_node(goal_node_id)

    first = _dijkstra_excluding(
        graph, start_node_id, goal_node_id, accessibility_mode, set(), set()
    )
    if first is None:
        return []

    found: list[Route] = [first]
    candidates: list[tuple[float, list[str], list[str]]] = []  # (distance, node_ids, edge_ids)
    seen_candidate_edge_sequences: set[tuple[str, ...]] = set()

    for _ in range(1, k):
        previous_route = found[-1]

        for i in range(len(previous_route.node_ids) - 1):
            spur_node = previous_route.node_ids[i]
            root_node_ids = previous_route.node_ids[: i + 1]
            root_edge_ids = previous_route.edge_ids[:i]

            excluded_edges: set[str] = set()
            for existing in found:
                if existing.node_ids[: i + 1] == root_node_ids and i < len(existing.edge_ids):
                    excluded_edges.add(existing.edge_ids[i])

            excluded_nodes = set(root_node_ids[:-1])  # every root node except the spur itself

            spur_route = _dijkstra_excluding(
                graph, spur_node, goal_node_id, accessibility_mode, excluded_nodes, excluded_edges
            )
            if spur_route is None:
                continue

            total_node_ids = root_node_ids[:-1] + spur_route.node_ids
            total_edge_ids = root_edge_ids + spur_route.edge_ids
            key = tuple(total_edge_ids)

            if key in seen_candidate_edge_sequences or any(tuple(r.edge_ids) == key for r in found):
                continue
            seen_candidate_edge_sequences.add(key)

            total_distance = sum(graph.get_edge(eid).distance_meters for eid in total_edge_ids)
            heapq.heappush(candidates, (total_distance, total_node_ids, total_edge_ids))

        if not candidates:
            break

        _, next_node_ids, next_edge_ids = heapq.heappop(candidates)
        found.append(
            Route.from_path(graph, next_node_ids, next_edge_ids, "yen_dijkstra", accessibility_mode)
        )

    return found


def _dijkstra_excluding(
    graph: CampusGraph,
    start_node_id: str,
    goal_node_id: str,
    accessibility_mode: AccessibilityMode,
    excluded_node_ids: set[str],
    excluded_edge_ids: set[str],
) -> Route | None:
    """Dijkstra variant that treats excluded nodes/edges as absent from the
    graph. Returns None (rather than raising) when no route exists, since
    "this spur path doesn't exist" is expected, routine control flow inside
    Yen's algorithm, not an application-level error."""
    if start_node_id == goal_node_id:
        return Route.from_path(graph, [start_node_id], [], "yen_dijkstra", accessibility_mode)

    distances: dict[str, float] = {start_node_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}
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
            if edge.id in excluded_edge_ids or neighbor.id in excluded_node_ids:
                continue

            weight = edge_traversal_weight(edge, accessibility_mode)
            if math.isinf(weight):
                continue

            candidate_distance = current_distance + weight
            if candidate_distance < distances.get(neighbor.id, math.inf):
                distances[neighbor.id] = candidate_distance
                previous[neighbor.id] = (current_node, edge.id)
                heapq.heappush(queue, (candidate_distance, neighbor.id))

    if goal_node_id not in distances:
        return None

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

    return Route.from_path(graph, node_ids, edge_ids, "yen_dijkstra", accessibility_mode)
