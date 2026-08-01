"""Tests for Dijkstra routing, including accessibility-mode behavior."""

import pytest

from campus_nav.core.exceptions import NodeNotFoundError, NoRouteFoundError
from campus_nav.navigation.routing.dijkstra import dijkstra_route
from campus_nav.navigation.routing.types import AccessibilityMode

pytestmark = pytest.mark.unit


def test_shortest_path_uses_stairs_shortcut_when_accessibility_ignored(simple_graph):
    route = dijkstra_route(simple_graph, "A", "C", AccessibilityMode.NONE)
    assert route.node_ids == ["A", "C"]
    assert route.total_distance_meters == pytest.approx(5.0)
    assert route.uses_inaccessible_edge is True


def test_strict_accessibility_avoids_stairs(simple_graph):
    route = dijkstra_route(simple_graph, "A", "C", AccessibilityMode.STRICT)
    assert route.node_ids == ["A", "B", "C"]
    assert route.total_distance_meters == pytest.approx(20.0)
    assert route.uses_inaccessible_edge is False


def test_preferred_accessibility_prefers_accessible_but_allows_fallback(simple_graph):
    # A->B->C (20m accessible) is preferred over A->C (5m, stairs) even
    # though the stairs route is shorter -- PREFERRED heavily penalizes it.
    route = dijkstra_route(simple_graph, "A", "C", AccessibilityMode.PREFERRED)
    assert route.node_ids == ["A", "B", "C"]


def test_same_start_and_goal_returns_trivial_route(simple_graph):
    route = dijkstra_route(simple_graph, "A", "A")
    assert route.node_ids == ["A"]
    assert route.edge_ids == []
    assert route.total_distance_meters == 0.0


def test_no_route_between_disconnected_components_raises(disconnected_graph):
    with pytest.raises(NoRouteFoundError):
        dijkstra_route(disconnected_graph, "A", "X")


def test_unknown_start_node_raises(simple_graph):
    with pytest.raises(NodeNotFoundError):
        dijkstra_route(simple_graph, "does-not-exist", "A")


def test_unknown_goal_node_raises(simple_graph):
    with pytest.raises(NodeNotFoundError):
        dijkstra_route(simple_graph, "A", "does-not-exist")


def test_route_algorithm_field_is_dijkstra(simple_graph):
    route = dijkstra_route(simple_graph, "A", "B")
    assert route.algorithm == "dijkstra"


def test_longer_but_only_route_when_strict_and_graph_becomes_disconnected():
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
    # Only route is inaccessible -- STRICT must report no route exists.
    with pytest.raises(NoRouteFoundError):
        dijkstra_route(graph, "A", "B", AccessibilityMode.STRICT)
