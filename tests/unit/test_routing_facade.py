"""Tests for the routing facade (find_route / find_accessible_route)."""

import pytest

from campus_nav.core.exceptions import NoRouteFoundError
from campus_nav.navigation.routing.facade import RoutingAlgorithm, find_accessible_route, find_route
from campus_nav.navigation.routing.types import AccessibilityMode

pytestmark = pytest.mark.unit


def test_find_route_defaults_to_astar(simple_graph):
    route = find_route(simple_graph, "A", "E")
    assert route.algorithm == "astar"


def test_find_route_can_select_dijkstra(simple_graph):
    route = find_route(simple_graph, "A", "E", algorithm=RoutingAlgorithm.DIJKSTRA)
    assert route.algorithm == "dijkstra"


def test_find_accessible_route_strict_avoids_stairs(simple_graph):
    route = find_accessible_route(simple_graph, "A", "C", strict=True)
    assert route.uses_inaccessible_edge is False


def test_find_accessible_route_strict_raises_when_impossible():
    from campus_nav.navigation.graph.models import CampusGraph, Edge, Node, PathType

    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    graph.add_node(Node(id="B", name="B", latitude=0, longitude=0))
    graph.add_edge(
        Edge(
            id="AB",
            from_node_id="A",
            to_node_id="B",
            distance_meters=10,
            path_type=PathType.STAIRS,
            is_accessible=False,
        )
    )
    with pytest.raises(NoRouteFoundError):
        find_accessible_route(graph, "A", "B", strict=True)


def test_find_accessible_route_non_strict_uses_preferred_mode(simple_graph):
    route = find_accessible_route(simple_graph, "A", "C", strict=False)
    assert route.accessibility_mode == AccessibilityMode.PREFERRED
