"""Tests for Yen's k-shortest-paths implementation (multi_route.py)."""

import pytest

from campus_nav.navigation.routing.multi_route import k_shortest_routes
from campus_nav.navigation.routing.types import AccessibilityMode

pytestmark = pytest.mark.unit


def test_returns_shortest_route_first(simple_graph):
    routes = k_shortest_routes(simple_graph, "A", "E", k=3)
    # A -stairs-> C -> E (5 + 10 = 15m) is the true shortest, via the
    # inaccessible shortcut edge included in the simple_graph fixture.
    assert routes[0].node_ids == ["A", "C", "E"]
    assert routes[0].total_distance_meters == pytest.approx(15.0)


def test_routes_are_sorted_by_increasing_distance(simple_graph):
    routes = k_shortest_routes(simple_graph, "A", "E", k=5)
    distances = [r.total_distance_meters for r in routes]
    assert distances == sorted(distances)


def test_routes_are_distinct(simple_graph):
    routes = k_shortest_routes(simple_graph, "A", "E", k=5)
    edge_sequences = [tuple(r.edge_ids) for r in routes]
    assert len(edge_sequences) == len(set(edge_sequences))


def test_requesting_more_routes_than_exist_returns_fewer(simple_graph):
    routes = k_shortest_routes(simple_graph, "A", "E", k=100)
    assert 1 <= len(routes) < 100


def test_k_equals_one_returns_single_shortest_route(simple_graph):
    routes = k_shortest_routes(simple_graph, "A", "E", k=1)
    assert len(routes) == 1


def test_no_route_returns_empty_list(disconnected_graph):
    routes = k_shortest_routes(disconnected_graph, "A", "X", k=3)
    assert routes == []


def test_accessibility_mode_respected_in_alternatives(simple_graph):
    routes = k_shortest_routes(
        simple_graph, "A", "C", k=5, accessibility_mode=AccessibilityMode.STRICT
    )
    assert all(not r.uses_inaccessible_edge for r in routes)


def test_pilot_graph_multiple_routes_between_gate_and_sncc(pilot_graph):
    routes = k_shortest_routes(pilot_graph, "main_gate", "sncc", k=3)
    assert len(routes) >= 2
    assert routes[0].total_distance_meters <= routes[-1].total_distance_meters
