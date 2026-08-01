"""Tests for A* routing, including cross-checking against Dijkstra for
optimality (the heuristic must never cause A* to miss the true shortest
path -- see astar.py docstring on heuristic admissibility)."""

import pytest

from campus_nav.core.exceptions import NodeNotFoundError, NoRouteFoundError
from campus_nav.navigation.routing.astar import astar_route
from campus_nav.navigation.routing.dijkstra import dijkstra_route
from campus_nav.navigation.routing.types import AccessibilityMode

pytestmark = pytest.mark.unit


def test_astar_matches_dijkstra_distance_on_simple_graph(simple_graph):
    astar = astar_route(simple_graph, "A", "E")
    dijkstra = dijkstra_route(simple_graph, "A", "E")
    assert astar.total_distance_meters == pytest.approx(dijkstra.total_distance_meters)


def test_astar_matches_dijkstra_on_pilot_graph_for_all_pairs(pilot_graph):
    node_ids = list(pilot_graph.nodes.keys())
    for i, start in enumerate(node_ids):
        for goal in node_ids[i + 1 :]:
            try:
                astar_result = astar_route(pilot_graph, start, goal)
            except NoRouteFoundError:
                astar_result = None
            try:
                dijkstra_result = dijkstra_route(pilot_graph, start, goal)
            except NoRouteFoundError:
                dijkstra_result = None

            if astar_result is None or dijkstra_result is None:
                assert astar_result is None and dijkstra_result is None
            else:
                assert astar_result.total_distance_meters == pytest.approx(
                    dijkstra_result.total_distance_meters
                )


def test_astar_strict_accessibility_avoids_stairs(simple_graph):
    route = astar_route(simple_graph, "A", "C", AccessibilityMode.STRICT)
    assert route.uses_inaccessible_edge is False
    assert route.node_ids == ["A", "B", "C"]


def test_astar_same_start_and_goal(simple_graph):
    route = astar_route(simple_graph, "A", "A")
    assert route.node_ids == ["A"]


def test_astar_no_route_raises(disconnected_graph):
    with pytest.raises(NoRouteFoundError):
        astar_route(disconnected_graph, "A", "X")


def test_astar_unknown_node_raises(simple_graph):
    with pytest.raises(NodeNotFoundError):
        astar_route(simple_graph, "ghost", "A")


def test_astar_algorithm_field(simple_graph):
    route = astar_route(simple_graph, "A", "B")
    assert route.algorithm == "astar"
