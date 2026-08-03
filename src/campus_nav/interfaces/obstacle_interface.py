"""
Obstacle event vocabulary -- the integration point between the Vision
Pipeline (Phase 1.3) and the Navigation Engine's Safety Engine (Phase 1.2).

Design rationale
-----------------
`ObstacleEvent` lives in `interfaces/`, not in `vision/`, specifically so
that `navigation.engine.safety_engine.SafetyEngine` can depend on this
shared vocabulary without importing anything from the `vision` package.
The dependency arrow points the same direction it already does for
localization: SafetyEngine depends on an abstract, stable value type
(compare `interfaces/localization_interface.py`'s `LocalizationResult`),
and the concrete producer (vision.safety.obstacle_events) depends on
SafetyEngine's additive `evaluate_obstacles` method, not the other way
around. This keeps `navigation/` free of any dependency on `vision/`,
which matters because navigation must keep working even when vision is
completely absent (no camera, optional [vision] extra not installed).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from campus_nav.vision.detection.categories import ObjectCategory


class ObstaclePosition(Enum):
    """Whether an obstacle is judged to be in the user's direct path or
    merely nearby. Matches the original specification's object detection
    policy: "beside path -> ignore", "blocking path -> warn"."""

    BLOCKING = "blocking"
    BESIDE_PATH = "beside_path"


@dataclass(frozen=True, slots=True)
class ObstacleEvent:
    """A single navigation-relevant obstacle observation, synthesized from
    one frame's detection + tracking + depth estimation results. This is
    the artifact `SafetyEngine.evaluate_obstacles` consumes, and also what
    the `/vision/obstacles` API endpoint reports -- and what a future
    Phase 1.4 Decision Engine will consume for richer fusion/reasoning."""

    track_id: int
    category: ObjectCategory
    label: str
    position: ObstaclePosition
    distance_meters: float | None
    """None if depth estimation was unavailable or failed for this object
    -- the event is still worth reporting (e.g. a door is always announced
    regardless of distance), just without a specific distance."""
    confidence: float
    announce_text: str | None
    """Pre-formatted, ready-to-speak text if this event warrants a safety
    announcement (e.g. "Obstacle ahead."), or None if this observation
    should be silently logged/tracked but not spoken -- e.g. a bench
    beside the path. Computed once by vision.safety.obstacle_events so
    SafetyEngine never needs its own copy of the object-category phrasing
    policy.
    """
