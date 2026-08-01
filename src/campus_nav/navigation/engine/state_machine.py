"""
Navigation state machine.

Design rationale
-----------------
The Phase 1.1 architecture review flagged that the original spec's Decision
Engine had no formal state model — an ad-hoc "fuses info" black box is
untestable. This class makes every allowed transition explicit and rejects
anything else, which means:

- Illegal sequences (e.g. `resume()` while already navigating) are caught
  immediately with a clear `InvalidStateTransitionError`, during
  development, rather than manifesting as confusing downstream behavior.
- The full transition table is visible in one place and unit-testable
  exhaustively (see tests/unit/test_state_machine.py), rather than scattered
  across conditional checks in the Navigation Engine.
"""

from __future__ import annotations

from campus_nav.core.exceptions import InvalidStateTransitionError
from campus_nav.interfaces.navigation_interface import NavEvent, NavState

# Maps (current_state, event) -> next_state. STOP is valid from every
# active state (handled specially below) so it isn't spelled out for each one.
_TRANSITIONS: dict[tuple[NavState, NavEvent], NavState] = {
    (NavState.IDLE, NavEvent.START): NavState.NAVIGATING,
    (NavState.NAVIGATING, NavEvent.PAUSE): NavState.PAUSED,
    (NavState.PAUSED, NavEvent.RESUME): NavState.NAVIGATING,
    (NavState.NAVIGATING, NavEvent.ARRIVE): NavState.ARRIVED,
    (NavState.NAVIGATING, NavEvent.LOSE_CONFIDENCE): NavState.UNCERTAIN,
    (NavState.UNCERTAIN, NavEvent.REGAIN_CONFIDENCE): NavState.NAVIGATING,
    (NavState.NAVIGATING, NavEvent.REROUTE_NEEDED): NavState.REROUTING,
    (NavState.UNCERTAIN, NavEvent.REROUTE_NEEDED): NavState.REROUTING,
    (NavState.REROUTING, NavEvent.REROUTE_COMPLETE): NavState.NAVIGATING,
}

# STOP is valid from any state except IDLE itself (stopping an already-idle
# session is a no-op the engine can choose to allow, but the state machine
# treats it as illegal so callers notice a redundant stop() call).
_STOP_ALLOWED_FROM = {
    NavState.NAVIGATING,
    NavState.PAUSED,
    NavState.UNCERTAIN,
    NavState.REROUTING,
    NavState.ARRIVED,
}


class NavigationStateMachine:
    """A minimal, explicit finite state machine for navigation safety
    state. Holds no navigation domain data (route, position) — purely the
    current NavState and the rules for changing it."""

    def __init__(self, initial_state: NavState = NavState.IDLE) -> None:
        self._state = initial_state

    @property
    def state(self) -> NavState:
        return self._state

    def can_transition(self, event: NavEvent) -> bool:
        if event is NavEvent.STOP:
            return self._state in _STOP_ALLOWED_FROM
        return (self._state, event) in _TRANSITIONS

    def transition(self, event: NavEvent) -> NavState:
        """Apply `event`, returning the new state. Raises
        InvalidStateTransitionError if the transition is not allowed from
        the current state."""
        if event is NavEvent.STOP:
            if self._state not in _STOP_ALLOWED_FROM:
                raise InvalidStateTransitionError(
                    f"Cannot STOP from state {self._state.value!r} "
                    f"(already idle or never started)."
                )
            self._state = NavState.IDLE
            return self._state

        key = (self._state, event)
        if key not in _TRANSITIONS:
            raise InvalidStateTransitionError(
                f"Event {event.value!r} is not valid from state {self._state.value!r}."
            )
        self._state = _TRANSITIONS[key]
        return self._state
