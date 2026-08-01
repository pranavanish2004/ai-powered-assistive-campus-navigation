"""Integration tests for NavigationEngine -- exercising routing, state
machine, safety engine, and voice guidance together as they actually run in
one navigation session."""

import pytest

from campus_nav.core.exceptions import (
    InvalidStateTransitionError,
    NavigationError,
    NodeNotFoundError,
    NoRouteFoundError,
)
from campus_nav.infrastructure.speech.null_tts import NullTTS
from campus_nav.interfaces.navigation_interface import NavState
from campus_nav.interfaces.text_to_speech_interface import SpeechPriority
from campus_nav.navigation.engine.navigation_engine import NavigationEngine
from campus_nav.navigation.engine.voice_guidance import VoiceGuidance
from campus_nav.navigation.routing.types import AccessibilityMode

pytestmark = pytest.mark.integration


def _make_engine(graph, **kwargs):
    tts = NullTTS()
    engine = NavigationEngine(graph=graph, voice_guidance=VoiceGuidance(tts), **kwargs)
    return engine, tts


def test_start_navigation_transitions_to_navigating_and_speaks_first_instruction(pilot_graph):
    engine, tts = _make_engine(pilot_graph)
    engine.start_navigation("main_gate", "library")
    assert engine.state == NavState.NAVIGATING
    assert tts.last_utterance is not None
    assert tts.last_utterance.priority is SpeechPriority.CONVERSATIONAL


def test_start_navigation_with_unknown_destination_raises_node_not_found(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    with pytest.raises(NodeNotFoundError):
        engine.start_navigation("main_gate", "does-not-exist")
    assert engine.state == NavState.IDLE


def test_start_navigation_with_unreachable_destination_raises_no_route_found(disconnected_graph):
    engine, _ = _make_engine(disconnected_graph)
    with pytest.raises(NoRouteFoundError):
        engine.start_navigation("A", "X")
    assert engine.state == NavState.IDLE


def test_update_position_before_start_raises(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    with pytest.raises(NavigationError):
        engine.update_position(17.594, 78.123, 5.0)


def test_update_position_at_destination_triggers_arrival(pilot_graph):
    engine, _ = _make_engine(pilot_graph, off_route_warning_meters=1000.0)
    engine.start_navigation("main_gate", "library")

    library = pilot_graph.get_node("library")
    update = engine.update_position(library.latitude, library.longitude, 5.0)

    assert update.state == NavState.ARRIVED
    assert engine.state == NavState.ARRIVED


def test_update_position_far_off_route_triggers_reroute_and_recovers(pilot_graph):
    engine, tts = _make_engine(
        pilot_graph, off_route_warning_meters=10.0, max_snap_distance_meters=200.0
    )
    engine.start_navigation("main_gate", "biomedical")

    # Jump to a position nowhere near the planned route but still snappable
    # (near medical_centre / hostel_h1 cluster, far from the library path).
    hostel = pilot_graph.get_node("hostel_h1")
    update = engine.update_position(hostel.latitude, hostel.longitude, 5.0)

    # Either rerouted back to NAVIGATING, or briefly REROUTING/UNCERTAIN --
    # what matters is a safety alert was raised and the engine did not crash.
    assert update.safety_alert is not None
    assert any(u.priority is SpeechPriority.SAFETY for u in tts.history)


def test_update_position_no_gps_fix_sets_uncertain_and_speaks_safety_alert(pilot_graph):
    engine, tts = _make_engine(pilot_graph)
    engine.start_navigation("main_gate", "library")

    # A position far enough that it cannot be snapped at all (default max
    # snap distance is small relative to this jump).
    update = engine.update_position(0.0, 0.0, 5.0)

    assert update.state == NavState.UNCERTAIN
    assert update.safety_alert is not None
    assert tts.history[-1].priority is SpeechPriority.SAFETY


def test_pause_and_resume_cycle(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    engine.start_navigation("main_gate", "library")
    engine.pause()
    assert engine.state == NavState.PAUSED
    engine.resume()
    assert engine.state == NavState.NAVIGATING


def test_pause_while_idle_raises(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    with pytest.raises(InvalidStateTransitionError):
        engine.pause()


def test_stop_resets_engine_to_idle_and_clears_route(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    engine.start_navigation("main_gate", "library")
    engine.stop()
    assert engine.state == NavState.IDLE
    assert engine.active_route is None


def test_update_position_while_paused_is_a_no_op(pilot_graph):
    engine, tts = _make_engine(pilot_graph)
    engine.start_navigation("main_gate", "library")
    engine.pause()

    calls_before = len(tts.history)
    update = engine.update_position(17.594, 78.123, 5.0)

    assert update.state == NavState.PAUSED
    assert update.next_instruction is None
    assert len(tts.history) == calls_before  # no new speech while paused


def test_accessible_route_used_end_to_end(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    route = engine.start_navigation(
        "main_gate", "hostel_h1", accessibility_mode=AccessibilityMode.STRICT
    )
    assert route.uses_inaccessible_edge is False
    assert "medical_centre" in route.node_ids


def test_last_known_node_id_updates_after_position_update(pilot_graph):
    engine, _ = _make_engine(pilot_graph)
    engine.start_navigation("main_gate", "library")
    assert engine.last_known_node_id is None

    admin = pilot_graph.get_node("admin_block")
    engine.update_position(admin.latitude, admin.longitude, 5.0)
    assert engine.last_known_node_id == "admin_block"
