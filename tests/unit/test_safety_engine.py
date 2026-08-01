"""Tests for the deterministic Safety Engine."""

import pytest

from campus_nav.interfaces.localization_interface import LocalizationResult
from campus_nav.interfaces.navigation_interface import NavEvent
from campus_nav.navigation.engine.safety_engine import SafetyEngine
from campus_nav.navigation.routing.dijkstra import dijkstra_route

pytestmark = pytest.mark.unit


def _localization_at(graph, node_id, confidence=1.0):
    node = graph.get_node(node_id)
    edges = graph.neighbors(node_id)
    return LocalizationResult(
        latitude=node.latitude,
        longitude=node.longitude,
        nearest_node=node,
        nearest_edge=edges[0][0] if edges else None,
        snap_distance_meters=0.0,
        confidence=confidence,
    )


def test_no_localization_triggers_lose_confidence(simple_graph):
    engine = SafetyEngine(off_route_warning_meters=15.0)
    route = dijkstra_route(simple_graph, "A", "E")
    decision = engine.evaluate(None, route, simple_graph)
    assert decision.event is NavEvent.LOSE_CONFIDENCE
    assert decision.alert_text is not None


def test_low_confidence_triggers_lose_confidence(simple_graph):
    engine = SafetyEngine(off_route_warning_meters=15.0, low_confidence_threshold=0.5)
    route = dijkstra_route(simple_graph, "A", "E")
    localization = _localization_at(simple_graph, "A", confidence=0.1)
    decision = engine.evaluate(localization, route, simple_graph)
    assert decision.event is NavEvent.LOSE_CONFIDENCE


def test_on_route_high_confidence_no_decision(simple_graph):
    engine = SafetyEngine(off_route_warning_meters=1000.0)
    route = dijkstra_route(simple_graph, "A", "E")
    localization = _localization_at(simple_graph, "B", confidence=1.0)
    decision = engine.evaluate(localization, route, simple_graph)
    assert decision.event is None
    assert decision.alert_text is None


def test_arrival_within_radius_triggers_arrive_event(simple_graph):
    engine = SafetyEngine(off_route_warning_meters=1000.0, arrival_radius_meters=5.0)
    route = dijkstra_route(simple_graph, "A", "E")
    localization = _localization_at(simple_graph, "E", confidence=1.0)
    decision = engine.evaluate(localization, route, simple_graph)
    assert decision.event is NavEvent.ARRIVE
    assert decision.alert_text is None


def test_off_route_drift_triggers_reroute_needed():
    from campus_nav.navigation.graph.models import CampusGraph, Edge, Node

    graph = CampusGraph()
    graph.add_node(Node(id="A", name="A", latitude=0.0, longitude=0.0))
    graph.add_node(Node(id="B", name="B", latitude=0.001, longitude=0.0))  # ~111m north
    graph.add_node(Node(id="Far", name="Far", latitude=0.01, longitude=0.01))  # far away
    graph.add_edge(Edge(id="AB", from_node_id="A", to_node_id="B", distance_meters=120))

    engine = SafetyEngine(off_route_warning_meters=15.0)
    route = dijkstra_route(graph, "A", "B")

    localization = LocalizationResult(
        latitude=graph.get_node("Far").latitude,
        longitude=graph.get_node("Far").longitude,
        nearest_node=graph.get_node("Far"),
        nearest_edge=None,
        snap_distance_meters=0.0,
        confidence=1.0,
    )
    decision = engine.evaluate(localization, route, graph)
    assert decision.event is NavEvent.REROUTE_NEEDED
    assert decision.alert_text is not None


def test_priority_order_arrival_checked_before_off_route(simple_graph):
    # Being exactly at the destination should report ARRIVE, not
    # REROUTE_NEEDED, even with an aggressive off-route threshold.
    engine = SafetyEngine(off_route_warning_meters=0.001, arrival_radius_meters=5.0)
    route = dijkstra_route(simple_graph, "A", "E")
    localization = _localization_at(simple_graph, "E", confidence=1.0)
    decision = engine.evaluate(localization, route, simple_graph)
    assert decision.event is NavEvent.ARRIVE
