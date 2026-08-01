"""Tests for GPS-to-graph snapping (nearest node/edge matching)."""

import pytest

from campus_nav.core.exceptions import SnappingFailedError
from campus_nav.navigation.localization.snapping import (
    find_nearest_edge,
    find_nearest_node,
    snap_to_graph,
)

pytestmark = pytest.mark.unit


def test_find_nearest_node_exact_match(simple_graph):
    node, distance = find_nearest_node(simple_graph, 17.590, 78.120)
    assert node.id == "A"
    assert distance == pytest.approx(0.0, abs=1.0)


def test_find_nearest_node_raises_on_empty_graph():
    from campus_nav.navigation.graph.models import CampusGraph

    with pytest.raises(SnappingFailedError):
        find_nearest_node(CampusGraph(), 0.0, 0.0)


def test_find_nearest_edge_returns_none_for_edgeless_graph():
    from campus_nav.navigation.graph.models import CampusGraph, Node

    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    assert find_nearest_edge(graph, 0.0, 0.0) is None


def test_find_nearest_edge_projects_onto_segment(simple_graph):
    # Roughly halfway between A and B geographically.
    result = find_nearest_edge(simple_graph, 17.5905, 78.1205)
    assert result is not None
    edge, proj_lat, proj_lon, distance = result
    assert edge.id in ("AB", "AC_stairs")  # both plausible near-A edges


def test_snap_to_graph_succeeds_within_max_distance(simple_graph):
    result = snap_to_graph(simple_graph, 17.590, 78.120, max_distance_meters=50.0)
    assert result.nearest_node.id == "A"
    assert result.distance_meters < 50.0


def test_snap_to_graph_raises_when_too_far(simple_graph):
    # Far away from the whole graph.
    with pytest.raises(SnappingFailedError):
        snap_to_graph(simple_graph, 0.0, 0.0, max_distance_meters=10.0)


def test_snap_to_graph_prefers_closer_of_node_or_edge(simple_graph):
    result = snap_to_graph(simple_graph, 17.590, 78.120, max_distance_meters=1000.0)
    # At node A's exact coordinate, the node match should win (distance ~0).
    assert result.distance_meters < 1.0
