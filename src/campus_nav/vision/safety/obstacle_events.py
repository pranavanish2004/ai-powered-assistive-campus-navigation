"""
Obstacle event derivation -- implements the project specification's
"Object Detection Policy" as deterministic, testable logic.

Policy recap (from the original specification):
  - Tree/bench/pole beside path -> ignore (no announcement)
  - Tree/bench/pole blocking path -> "Obstacle ahead" (+ direction hint)
  - Door -> "The entrance is directly ahead" (always, regardless of position)
  - Ramp -> "Ramp ahead" (always)
  - Stairs -> "Stairs ahead" (always)
  - Person approaching -> "Someone is approaching"
  - Everything else: silent unless blocking

This module produces `ObstacleEvent` objects (interfaces/obstacle_interface.py)
from one frame's `VisionFrameResult` plus depth estimates -- pure functions,
no I/O, no speech. `vision.safety.vision_safety_bridge` (not this module)
is responsible for actually deciding whether/when to speak an event; this
module only decides *what a given observation means* and pre-formats the
announcement text a caller may choose to use.
"""

from __future__ import annotations

from campus_nav.interfaces.detection_interface import BoundingBox
from campus_nav.interfaces.obstacle_interface import ObstacleEvent, ObstaclePosition
from campus_nav.interfaces.tracking_interface import TrackedObject
from campus_nav.vision.detection.categories import (
    PERSON_CATEGORIES,
    ObjectCategory,
    is_always_announce,
)
from campus_nav.vision.pipeline.frame_result import VisionFrameResult

# An object is considered to be "in the path" if its bounding-box center
# falls within this fraction of the frame's width, centered on the middle
# of the frame -- e.g. 0.5 means the middle 50% of the frame width.
_PATH_CENTER_WIDTH_FRACTION = 0.5

_ALWAYS_ANNOUNCE_TEXT: dict[ObjectCategory, str] = {
    ObjectCategory.DOOR: "The entrance is directly ahead.",
    ObjectCategory.RAMP: "Ramp ahead.",
    ObjectCategory.STAIRS: "Stairs ahead.",
}

_BLOCKING_TEXT_DEFAULT = "Obstacle ahead."


def classify_position(bbox: BoundingBox, frame_width: int) -> ObstaclePosition:
    """Classify whether an object's bounding box falls within the user's
    walking path (roughly the horizontal center of the frame) or off to
    one side."""
    if frame_width <= 0:
        return ObstaclePosition.BESIDE_PATH

    half_band = (frame_width * _PATH_CENTER_WIDTH_FRACTION) / 2.0
    center = frame_width / 2.0
    return (
        ObstaclePosition.BLOCKING
        if (center - half_band) <= bbox.center_x <= (center + half_band)
        else ObstaclePosition.BESIDE_PATH
    )


def _announcement_for(
    tracked: TrackedObject,
    position: ObstaclePosition,
    distance_meters: float | None,
    safe_distance_meters: float,
) -> str | None:
    if is_always_announce(tracked.category):
        return _ALWAYS_ANNOUNCE_TEXT.get(tracked.category, _BLOCKING_TEXT_DEFAULT)

    if tracked.category in PERSON_CATEGORIES:
        # A person is only worth announcing when close enough to matter --
        # "someone is approaching" implies proximity, not merely presence
        # anywhere in frame.
        if distance_meters is not None and distance_meters <= safe_distance_meters:
            return "Someone is approaching."
        return None

    # Conditional categories (and UNKNOWN): silent unless actually blocking
    # AND close enough to be a real, near-term hazard -- a bench blocking
    # the path 10 meters away is not yet actionable information.
    if position is ObstaclePosition.BLOCKING and (
        distance_meters is None or distance_meters <= safe_distance_meters
    ):
        return _BLOCKING_TEXT_DEFAULT

    return None


def build_obstacle_events(
    frame_result: VisionFrameResult, safe_distance_meters: float
) -> list[ObstacleEvent]:
    """Derive `ObstacleEvent`s for every currently tracked object in
    `frame_result`, applying the object detection policy. Returns one
    event per tracked object -- including ones with `announce_text=None`,
    since "this object exists and is being tracked but isn't worth
    mentioning" is itself useful information for the `/vision/obstacles`
    diagnostic endpoint and the future Decision Engine, not something to
    discard.
    """
    events: list[ObstacleEvent] = []

    for tracked in frame_result.tracked_objects:
        depth = frame_result.depth_estimates.get(tracked.track_id)
        distance_meters = depth.distance_meters if depth else None
        confidence = depth.confidence if depth else tracked.confidence

        position = classify_position(tracked.bbox, frame_result.frame_width)
        announce_text = _announcement_for(tracked, position, distance_meters, safe_distance_meters)

        events.append(
            ObstacleEvent(
                track_id=tracked.track_id,
                category=tracked.category,
                label=tracked.label,
                position=position,
                distance_meters=distance_meters,
                confidence=confidence,
                announce_text=announce_text,
            )
        )

    return events
