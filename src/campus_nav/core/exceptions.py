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


# ===========================================================================
# Vision Pipeline (Phase 1.3)
# ===========================================================================
class DetectionError(RecoverableError):
    """Object detection failed on a given frame (e.g. malformed frame,
    inference backend error). Recoverable: the vision pipeline should skip
    the frame and continue, not crash navigation — obstacle detection is a
    supplementary safety signal, not the primary navigation mechanism."""


class OCRError(RecoverableError):
    """OCR failed to process a frame or region of interest. Recoverable:
    the assistant should simply have no text to report for this frame."""


class DepthEstimationError(RecoverableError):
    """Monocular depth estimation failed for a detected object. Recoverable:
    the object can still be reported without a distance estimate, with
    lower confidence."""


class TrackingError(RecoverableError):
    """The object tracker failed to update (e.g. inconsistent detection
    input). Recoverable: tracking can reset and resume from the next frame
    without losing overall vision pipeline operation."""


class ModelLoadError(RecoverableError):
    """A vision model (YOLO weights, OCR model, depth model) failed to
    load. Recoverable rather than fatal: per the Phase 1.1 architecture
    review, vision is a supplementary safety signal — navigation via the
    route/state machine must continue even if vision is entirely
    unavailable, with the Vision Pipeline reporting itself as unavailable
    rather than taking down the whole application."""


# ===========================================================================
# Dashboard Integration (Phase 1.4)
# ===========================================================================
class DashboardError(RecoverableError):
    """A campus dashboard data source (e.g. the IITH Dashboard API) failed
    to respond, timed out, or returned malformed data. Recoverable by
    design: per the original specification, "the system must work even if
    these APIs are unavailable," and per the Phase 1.4 decision to treat
    the dashboard as a replaceable, non-core data source — the assistant
    must continue functioning on the Campus Graph, GPS, Vision, OCR, and
    RAG/local knowledge base alone if this fails."""


# ===========================================================================
# RAG / Knowledge Base (Phase 1.4a)
# ===========================================================================
class RAGError(RecoverableError):
    """The knowledge base failed to load or search. Recoverable: the
    Decision Engine should proceed without retrieved context (and say so
    honestly) rather than fail the entire conversational response."""


# ===========================================================================
# MLLM (Phase 1.4c)
# ===========================================================================
class MLLMError(RecoverableError):
    """The configured MLLM provider failed to generate a response (network
    failure, API error, missing credentials, provider SDK not installed).
    Recoverable: per the two-lane safety architecture established since
    Phase 1.1, the MLLM only ever serves the CONVERSATIONAL lane — its
    failure must never affect navigation, obstacle warnings, or any
    SAFETY-priority speech, all of which are deterministic and MLLM-free.
    The Decision Engine falls back to a plain, honest "I can't answer that
    right now" rather than crashing or hallucinating a response."""


# ===========================================================================
# Decision Engine (Phase 1.4d)
# ===========================================================================
class DecisionEngineError(RecoverableError):
    """The Decision Engine failed to assemble or process a conversational
    request. Recoverable: a failed "ask AI" query must never affect active
    navigation or safety-critical obstacle detection, which run entirely
    independently of this code path."""


# ===========================================================================
# Campus Data Ingestion (Phase 1.4 refinement)
# ===========================================================================
class IngestionError(RecoverableError):
    """A campus data importer failed to parse a source file (malformed
    GeoJSON/CSV/YAML/Markdown, missing required fields, invalid
    coordinates). Recoverable: the ingestion pipeline skips the offending
    file and logs a clear warning rather than corrupting the Navigation
    Graph or knowledge base with partial or incorrect data -- silently
    accepting a malformed coordinate in a navigation-for-the-blind system
    is a safety concern, not just a data-quality one."""
