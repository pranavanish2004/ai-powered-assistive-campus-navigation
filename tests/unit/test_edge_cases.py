"""Miscellaneous edge-case tests that don't fit neatly under one module --
empty graphs, single-node graphs, zero-length routes, boundary confidence
values, etc. Complementary to the edge cases already embedded in each
module's own test file."""

import pytest

from campus_nav.core.exceptions import NodeNotFoundError, NoRouteFoundError, SnappingFailedError
from campus_nav.navigation.graph.models import CampusGraph, Node
from campus_nav.navigation.localization.confidence import confidence_from_snap_distance
from campus_nav.navigation.localization.snapping import snap_to_graph
from campus_nav.navigation.routing.dijkstra import dijkstra_route
from campus_nav.navigation.routing.multi_route import k_shortest_routes

pytestmark = pytest.mark.unit


def test_single_node_graph_route_to_self():
    graph = CampusGraph()
    graph.add_node(Node(id="only", name="Only Node", latitude=0, longitude=0))
    route = dijkstra_route(graph, "only", "only")
    assert route.node_ids == ["only"]
    assert route.is_empty() is True  # <2 nodes: no meaningful path to instruct on
    assert route.total_distance_meters == 0.0


def test_single_node_graph_route_to_nonexistent_node_raises():
    graph = CampusGraph()
    graph.add_node(Node(id="only", name="Only Node", latitude=0, longitude=0))
    with pytest.raises(NodeNotFoundError):
        dijkstra_route(graph, "only", "ghost")


def test_empty_graph_snap_raises_snapping_failed():
    graph = CampusGraph()
    with pytest.raises(SnappingFailedError):
        snap_to_graph(graph, 0.0, 0.0, max_distance_meters=100.0)


def test_k_shortest_routes_on_single_node_graph():
    graph = CampusGraph()
    graph.add_node(Node(id="only", name="Only", latitude=0, longitude=0))
    routes = k_shortest_routes(graph, "only", "only", k=3)
    assert len(routes) == 1
    assert routes[0].node_ids == ["only"]


def test_negative_snap_distance_never_produces_confidence_above_one():
    # distance_meters could theoretically be 0 exactly; confidence must be
    # clamped at 1.0, never exceed it even with floating point noise.
    assert confidence_from_snap_distance(-1.0, 30.0) <= 1.0


def test_route_is_empty_true_only_for_fewer_than_two_nodes():
    from campus_nav.navigation.routing.types import AccessibilityMode, Route

    single_node_route = Route(
        node_ids=["a"],
        edge_ids=[],
        total_distance_meters=0.0,
        algorithm="dijkstra",
        accessibility_mode=AccessibilityMode.NONE,
        uses_inaccessible_edge=False,
    )
    two_node_route = Route(
        node_ids=["a", "b"],
        edge_ids=["ab"],
        total_distance_meters=10.0,
        algorithm="dijkstra",
        accessibility_mode=AccessibilityMode.NONE,
        uses_inaccessible_edge=False,
    )
    assert single_node_route.is_empty() is True
    assert two_node_route.is_empty() is False


def test_graph_with_only_isolated_nodes_has_no_routes(disconnected_graph):
    with pytest.raises(NoRouteFoundError):
        dijkstra_route(disconnected_graph, "A", "Y")


def test_edge_weight_infinite_for_strict_inaccessible_edge():
    import math

    from campus_nav.navigation.graph.models import Edge, PathType
    from campus_nav.navigation.routing.types import AccessibilityMode, edge_traversal_weight

    edge = Edge(
        id="e1",
        from_node_id="a",
        to_node_id="b",
        distance_meters=10.0,
        path_type=PathType.STAIRS,
        is_accessible=False,
    )
    assert math.isinf(edge_traversal_weight(edge, AccessibilityMode.STRICT))
    assert edge_traversal_weight(edge, AccessibilityMode.NONE) == 10.0
    assert edge_traversal_weight(edge, AccessibilityMode.PREFERRED) > 10.0
