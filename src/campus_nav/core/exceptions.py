"""
Application-wide exception hierarchy.

Design rationale
-----------------
For a voice-guided assistant used by someone who cannot see a stack trace,
every error must be classifiable by *what the system should do about it*,
not just what broke. The hierarchy has exactly two direct branches under
`CampusNavError`:

- `RecoverableError`: the calling module should retry, fall back to a
  degraded mode, or substitute a cached/estimated value. The user may hear
  nothing, or a soft "reconnecting..." style message.
- `FatalError`: the calling module CANNOT safely continue this operation.
  The Navigation/Decision Engine must be informed so it can move the user
  into an "uncertain" safety state rather than silently failing.

Every leaf exception inherits from *exactly one* of these two branches
(never both) — this is a deliberate constraint that avoids the MRO
(Method Resolution Order) conflicts that a naive multi-inheritance design
runs into when two branches sharing a common ancestor are combined on the
same class. (An earlier version of this hierarchy hit exactly this bug —
`class Foo(CampusNavError, RecoverableError)` — since `RecoverableError`
already inherits from `CampusNavError`, Python cannot form a consistent
MRO. The rule "exactly one branch per leaf" prevents that class of bug by
construction, and is enforced by `test_exceptions.py`.)

Domain-specific subgroups (Device, Database, Graph, Localization,
Navigation) exist purely for `except` specificity — e.g. code that only
cares about graph problems can catch `GraphError` without also catching
unrelated database errors — but they do NOT sit between `CampusNavError`
and the Recoverable/Fatal branch, precisely to keep the MRO simple.
"""

from __future__ import annotations


class CampusNavError(Exception):
    """Base class for all application-specific exceptions."""


class RecoverableError(CampusNavError):
    """The operation failed but the system can safely retry, fall back, or
    degrade gracefully without endangering the user."""


class FatalError(CampusNavError):
    """The operation failed in a way that must not be silently swallowed —
    the Navigation/Decision Engine must be notified to change safety state."""


# ===========================================================================
# Configuration
# ===========================================================================
class ConfigurationError(FatalError):
    """Invalid or missing configuration. Should only ever occur at startup,
    since Settings validation is meant to fail fast before the app serves
    any requests."""


# ===========================================================================
# Device / hardware interfaces
# ===========================================================================
class CameraError(RecoverableError):
    """Camera could not be opened or a frame could not be read. Recoverable:
    the vision pipeline should fall back to the last known frame / audio-only
    guidance rather than crashing navigation."""


class CameraUnavailableError(FatalError):
    """Camera is unavailable for an extended period. This IS fatal for the
    vision pipeline (not the whole app) because obstacle detection cannot
    run blind — the Decision Engine must warn the user explicitly."""


class GPSError(RecoverableError):
    """GPS fix unavailable (e.g. indoors). Recoverable: navigation should
    fall back to landmark/last-known-position localization, not crash."""


# ===========================================================================
# Speech
# ===========================================================================
class SpeechRecognitionError(RecoverableError):
    """Speech could not be transcribed (e.g. unintelligible audio). The
    assistant should ask the user to repeat themselves, not fail silently."""


class TextToSpeechError(FatalError):
    """TTS engine failed to produce audio output. Fatal in the sense that
    the assistant has no way to communicate with the user until resolved."""


# ===========================================================================
# Database
# ===========================================================================
class DatabaseError(FatalError):
    """Database connection or query failure."""


# ===========================================================================
# Campus Graph (Phase 1.2)
# ===========================================================================
class GraphError(RecoverableError):
    """Base class for campus graph problems that a caller might reasonably
    recover from (e.g. by trying a cached/previous graph version)."""


class GraphValidationError(FatalError):
    """The campus graph data itself is structurally invalid (dangling edge
    references, duplicate node ids, disconnected required nodes, etc.).
    This is fatal because routing over an invalid graph could silently
    produce an unsafe or nonsensical route — the system must refuse to load
    it rather than guess."""


class NodeNotFoundError(GraphError):
    """A referenced node id does not exist in the graph."""


class EdgeNotFoundError(GraphError):
    """A referenced edge id does not exist in the graph."""


class GraphSerializationError(RecoverableError):
    """The graph could not be serialized to or deserialized from its
    on-disk representation (e.g. malformed JSON)."""


# ===========================================================================
# Routing (Phase 1.2)
# ===========================================================================
class RoutingError(RecoverableError):
    """Base class for routing failures."""


class NoRouteFoundError(RoutingError):
    """No path exists between the requested origin and destination — e.g.
    the graph is disconnected, or every path is blocked by accessibility
    constraints. Recoverable: the Navigation Engine should inform the user
    and offer alternatives rather than crash."""


# ===========================================================================
# Localization (Phase 1.2)
# ===========================================================================
class LocalizationError(RecoverableError):
    """Base class for localization failures (no fix, snapping failed)."""


class SnappingFailedError(LocalizationError):
    """No graph node/edge was found within the configured maximum snapping
    distance of the raw position fix. Recoverable: the Navigation Engine
    should widen its uncertainty rather than trust a wildly wrong snap."""


# ===========================================================================
# Navigation (Phase 1.2)
# ===========================================================================
class NavigationError(RecoverableError):
    """Base class for navigation orchestration failures."""


class InvalidStateTransitionError(FatalError):
    """The Navigation state machine was asked to perform a transition that
    is not valid from its current state (e.g. 'resume' while already
    navigating). Fatal because it indicates a logic bug in the caller, not
    an environmental condition the user caused — it should surface loudly
    during development/testing rather than be silently ignored."""
