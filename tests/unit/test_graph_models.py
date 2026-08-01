"""Tests for CampusGraph, Node, Edge -- construction, queries, and
serialization round-tripping."""

import pytest

from campus_nav.core.exceptions import EdgeNotFoundError, GraphValidationError, NodeNotFoundError
from campus_nav.navigation.graph.models import CampusGraph, Edge, Node, NodeType, PathType

pytestmark = pytest.mark.unit


def test_add_node_and_get_node(simple_graph):
    node = simple_graph.get_node("A")
    assert node.name == "Node A"


def test_get_missing_node_raises():
    graph = CampusGraph()
    with pytest.raises(NodeNotFoundError):
        graph.get_node("does-not-exist")


def test_get_missing_edge_raises(simple_graph):
    with pytest.raises(EdgeNotFoundError):
        simple_graph.get_edge("does-not-exist")


def test_duplicate_node_id_rejected():
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    with pytest.raises(GraphValidationError):
        graph.add_node(Node(id="A", name="A duplicate", latitude=1, longitude=1))


def test_duplicate_edge_id_rejected():
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    graph.add_node(Node(id="B", name="B", latitude=0, longitude=0))
    graph.add_edge(Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=10))
    with pytest.raises(GraphValidationError):
        graph.add_edge(Edge(id="AB", from_node_id="B", to_node_id="A", distance_meters=5))


def test_edge_with_unknown_from_node_rejected():
    graph = CampusGraph()
    graph.add_node(Node(id="B", name="B", latitude=0, longitude=0))
    with pytest.raises(GraphValidationError):
        graph.add_edge(Edge(id="AB", from_node_id="missing", to_node_id="B", distance_meters=10))


def test_edge_with_unknown_to_node_rejected():
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    with pytest.raises(GraphValidationError):
        graph.add_edge(Edge(id="AB", from_node_id="A", to_node_id="missing", distance_meters=10))


def test_edge_with_non_positive_distance_rejected():
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    graph.add_node(Node(id="B", name="B", latitude=0, longitude=0))
    with pytest.raises(GraphValidationError):
        graph.add_edge(Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=0))


def test_neighbors_bidirectional_edge_visible_from_both_ends(simple_graph):
    neighbors_of_a = {n.id for _, n in simple_graph.neighbors("A")}
    neighbors_of_b = {n.id for _, n in simple_graph.neighbors("B")}
    assert "B" in neighbors_of_a
    assert "A" in neighbors_of_b


def test_neighbors_one_way_edge_only_visible_from_source():
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    graph.add_node(Node(id="B", name="B", latitude=0, longitude=0))
    graph.add_edge(
        Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=10, bidirectional=False)
    )
    assert "B" in {n.id for _, n in graph.neighbors("A")}
    assert "A" not in {n.id for _, n in graph.neighbors("B")}


def test_neighbors_of_unknown_node_raises():
    graph = CampusGraph()
    with pytest.raises(NodeNotFoundError):
        graph.neighbors("nonexistent")


def test_node_count_and_edge_count(simple_graph):
    assert simple_graph.node_count() == 5
    assert simple_graph.edge_count() == 6


def test_edge_other_end_returns_correct_node():
    edge = Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=10)
    assert edge.other_end("A") == "B"
    assert edge.other_end("B") == "A"


def test_edge_other_end_raises_for_non_endpoint():
    edge = Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=10)
    with pytest.raises(ValueError):
        edge.other_end("C")


def test_node_to_dict_and_from_dict_round_trip():
    node = Node(
        id="n1",
        name="Test Node",
        latitude=1.5,
        longitude=2.5,
        node_type=NodeType.LANDMARK,
        floor=2,
        is_accessible=False,
        landmark_description="A test landmark.",
    )
    restored = Node.from_dict(node.to_dict())
    assert restored == node


def test_edge_to_dict_and_from_dict_round_trip():
    edge = Edge(
        id="e1",
        from_node_id="a",
        to_node_id="b",
        distance_meters=42.0,
        path_type=PathType.RAMP,
        is_accessible=True,
        bidirectional=False,
        landmark_description="A test edge.",
    )
    restored = Edge.from_dict(edge.to_dict())
    assert restored == edge


def test_graph_to_dict_and_from_dict_round_trip(simple_graph):
    data = simple_graph.to_dict()
    restored = CampusGraph.from_dict(data)
    assert restored.node_count() == simple_graph.node_count()
    assert restored.edge_count() == simple_graph.edge_count()
    assert restored.get_node("A").name == "Node A"


def test_validate_reports_isolated_nodes():
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0, longitude=0))
    graph.add_node(Node(id="isolated", name="Isolated", latitude=1, longitude=1))
    warnings = graph.validate()
    assert any("isolated" in w for w in warnings)


def test_validate_returns_no_warnings_for_fully_connected_graph(simple_graph):
    assert simple_graph.validate() == []
