"""Tests for SafetyEngine.evaluate_obstacles -- the Phase 1.3 integration
point feeding vision-derived obstacle observations into the existing
(Phase 1.2) Safety Engine, without touching evaluate()'s existing
behavior (see test_safety_engine.py for those unchanged tests)."""

import pytest

from campus_nav.interfaces.obstacle_interface import ObstacleEvent, ObstaclePosition
from campus_nav.navigation.engine.safety_engine import SafetyEngine
from campus_nav.vision.detection.categories import ObjectCategory

pytestmark = pytest.mark.unit


def _event(announce_text, track_id=1, category=ObjectCategory.BENCH):
    return ObstacleEvent(
        track_id=track_id,
        category=category,
        label=category.value,
        position=ObstaclePosition.BLOCKING,
        distance_meters=1.0,
        confidence=0.8,
        announce_text=announce_text,
    )


def test_no_events_produces_no_decision():
    engine = SafetyEngine(off_route_warning_meters=15.0)
    decision = engine.evaluate_obstacles([])
    assert decision.alert_text is None
    assert decision.event is None


def test_events_with_no_announcement_produce_no_decision():
    engine = SafetyEngine(off_route_warning_meters=15.0)
    decision = engine.evaluate_obstacles([_event(None), _event(None, track_id=2)])
    assert decision.alert_text is None


def test_first_blocking_event_is_announced():
    engine = SafetyEngine(off_route_warning_meters=15.0)
    decision = engine.evaluate_obstacles([_event("Obstacle ahead.")])
    assert decision.alert_text == "Obstacle ahead."
    assert decision.event is None  # obstacle events do not drive state transitions in Phase 1.3


def test_silent_events_are_skipped_in_favor_of_the_first_announceable_one():
    engine = SafetyEngine(off_route_warning_meters=15.0)
    events = [
        _event(None, track_id=1),
        _event("Stairs ahead.", track_id=2, category=ObjectCategory.STAIRS),
        _event("Obstacle ahead.", track_id=3),
    ]
    decision = engine.evaluate_obstacles(events)
    assert decision.alert_text == "Stairs ahead."


def test_evaluate_obstacles_does_not_affect_existing_evaluate_method(simple_graph):
    """Regression guard: adding evaluate_obstacles() must not change the
    behavior of the pre-existing evaluate() method used by Phase 1.2."""
    from campus_nav.navigation.routing.dijkstra import dijkstra_route

    engine = SafetyEngine(off_route_warning_meters=1000.0)
    route = dijkstra_route(simple_graph, "A", "E")

    decision = engine.evaluate(None, route, simple_graph)
    assert decision.alert_text is not None  # unchanged: no-localization behavior intact
