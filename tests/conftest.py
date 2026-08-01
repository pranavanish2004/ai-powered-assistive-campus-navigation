"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# NOTE: do NOT set APP_ENV here. pydantic-settings reads real environment
# variables regardless of a test passing `Settings(_env_file=None)` --
# that argument only disables loading a .env *file*, it does not stop
# BaseSettings from reading os.environ. Setting APP_ENV globally here would
# permanently override the "development" field default for every Settings
# instance constructed anywhere in the test session (this previously broke
# tests/unit/test_config.py::test_settings_load_with_defaults, which
# asserts the true, unmodified default).
os.environ.setdefault("LOG_DIR", "logs_test")

from campus_nav.navigation.graph.models import CampusGraph, Edge, Node, PathType

FIXTURE_GRAPH_PATH = Path(__file__).parent / "fixtures" / "sample_campus_graph.json"


@pytest.fixture
def simple_graph() -> CampusGraph:
    """A small, hand-built graph for fast algorithm-level unit tests.

    Layout (all distances in meters):

        A --10-- B --10-- C
        |                 |
        20                10
        |                 |
        D ------30------- E

    Plus a direct, inaccessible shortcut A -> C (stairs, 5m) so
    accessibility-mode tests have something meaningful to exclude.
    """
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="Node A", latitude=17.590, longitude=78.120))
    graph.add_node(Node(id="B", name="Node B", latitude=17.591, longitude=78.121))
    graph.add_node(Node(id="C", name="Node C", latitude=17.592, longitude=78.122))
    graph.add_node(Node(id="D", name="Node D", latitude=17.589, longitude=78.121))
    graph.add_node(Node(id="E", name="Node E", latitude=17.591, longitude=78.123))

    graph.add_edge(Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=10))
    graph.add_edge(Edge(id="BC", from_node_id="B", to_node_id="C", distance_meters=10))
    graph.add_edge(Edge(id="AD", from_node_id="A", to_node_id="D", distance_meters=20))
    graph.add_edge(Edge(id="CE", from_node_id="C", to_node_id="E", distance_meters=10))
    graph.add_edge(Edge(id="DE", from_node_id="D", to_node_id="E", distance_meters=30))
    graph.add_edge(
        Edge(
            id="AC_stairs",
            from_node_id="A",
            to_node_id="C",
            distance_meters=5,
            path_type=PathType.STAIRS,
            is_accessible=False,
        )
    )
    return graph


@pytest.fixture
def disconnected_graph() -> CampusGraph:
    """Two separate components: A-B and X-Y, with no edge between them."""
    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=17.590, longitude=78.120))
    graph.add_node(Node(id="B", name="B", latitude=17.591, longitude=78.121))
    graph.add_node(Node(id="X", name="X", latitude=17.600, longitude=78.130))
    graph.add_node(Node(id="Y", name="Y", latitude=17.601, longitude=78.131))
    graph.add_edge(Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=10))
    graph.add_edge(Edge(id="XY", from_node_id="X", to_node_id="Y", distance_meters=10))
    return graph


@pytest.fixture
def pilot_graph() -> CampusGraph:
    """The IIT Hyderabad pilot fixture graph, loaded fresh for each test."""
    from campus_nav.navigation.graph.serialization import load_graph_json

    return load_graph_json(FIXTURE_GRAPH_PATH)
