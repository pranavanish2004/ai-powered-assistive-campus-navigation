"""Exhaustive tests for the navigation state machine's transition table."""

import pytest

from campus_nav.core.exceptions import InvalidStateTransitionError
from campus_nav.interfaces.navigation_interface import NavEvent, NavState
from campus_nav.navigation.engine.state_machine import NavigationStateMachine

pytestmark = pytest.mark.unit


def test_initial_state_is_idle():
    sm = NavigationStateMachine()
    assert sm.state == NavState.IDLE


def test_start_transitions_idle_to_navigating():
    sm = NavigationStateMachine()
    assert sm.transition(NavEvent.START) == NavState.NAVIGATING


def test_full_happy_path_lifecycle():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    assert sm.state == NavState.NAVIGATING
    sm.transition(NavEvent.PAUSE)
    assert sm.state == NavState.PAUSED
    sm.transition(NavEvent.RESUME)
    assert sm.state == NavState.NAVIGATING
    sm.transition(NavEvent.ARRIVE)
    assert sm.state == NavState.ARRIVED
    sm.transition(NavEvent.STOP)
    assert sm.state == NavState.IDLE


def test_uncertain_and_recovery_cycle():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    sm.transition(NavEvent.LOSE_CONFIDENCE)
    assert sm.state == NavState.UNCERTAIN
    sm.transition(NavEvent.REGAIN_CONFIDENCE)
    assert sm.state == NavState.NAVIGATING


def test_reroute_cycle_from_navigating():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    sm.transition(NavEvent.REROUTE_NEEDED)
    assert sm.state == NavState.REROUTING
    sm.transition(NavEvent.REROUTE_COMPLETE)
    assert sm.state == NavState.NAVIGATING


def test_reroute_cycle_from_uncertain():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    sm.transition(NavEvent.LOSE_CONFIDENCE)
    sm.transition(NavEvent.REROUTE_NEEDED)
    assert sm.state == NavState.REROUTING


@pytest.mark.parametrize(
    "start_state",
    [
        NavState.NAVIGATING,
        NavState.PAUSED,
        NavState.UNCERTAIN,
        NavState.REROUTING,
        NavState.ARRIVED,
    ],
)
def test_stop_is_valid_from_every_active_state(start_state):
    sm = NavigationStateMachine(initial_state=start_state)
    assert sm.transition(NavEvent.STOP) == NavState.IDLE


def test_stop_from_idle_raises():
    sm = NavigationStateMachine(initial_state=NavState.IDLE)
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(NavEvent.STOP)


def test_resume_while_already_navigating_raises():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(NavEvent.RESUME)


def test_pause_while_idle_raises():
    sm = NavigationStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(NavEvent.PAUSE)


def test_start_while_already_navigating_raises():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(NavEvent.START)


def test_arrive_while_paused_raises():
    sm = NavigationStateMachine()
    sm.transition(NavEvent.START)
    sm.transition(NavEvent.PAUSE)
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(NavEvent.ARRIVE)


def test_can_transition_does_not_mutate_state():
    sm = NavigationStateMachine()
    assert sm.can_transition(NavEvent.START) is True
    assert sm.can_transition(NavEvent.PAUSE) is False
    assert sm.state == NavState.IDLE  # unchanged


def test_illegal_transition_leaves_state_unchanged():
    sm = NavigationStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(NavEvent.PAUSE)
    assert sm.state == NavState.IDLE
