"""
Object tracking abstraction.

Assigns persistent identity to detections across frames -- without
tracking, every frame's detections are independent, so a bench detected in
frame N and the same bench in frame N+1 look like two different objects.
Persistent ids are what let the vision pipeline reason about "is this the
object I already warned about" and smooth out single-frame detection noise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from campus_nav.interfaces.detection_interface import BoundingBox, Detection
from campus_nav.vision.detection.categories import ObjectCategory


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """A detection that has been assigned a persistent track id and
    (optionally) smoothed across recent frames."""

    track_id: int
    category: ObjectCategory
    label: str
    confidence: float
    bbox: BoundingBox
    frames_tracked: int
    """How many consecutive frames this track has been matched to a
    detection -- a track with frames_tracked=1 just appeared and may still
    be noise; a track with frames_tracked=10 is well-established."""
    frames_missed: int
    """Consecutive frames since the last matching detection (0 if matched
    this frame). Used to handle brief occlusions: a track survives with
    frames_missed > 0 up to a configured maximum before being dropped."""


class ObjectTrackerInterface(ABC):
    """Abstract base class for multi-object tracking backends."""

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Advance the tracker by one frame given this frame's raw
        detections, returning the current set of tracked objects (including
        any tracks temporarily unmatched this frame but still within their
        occlusion-tolerance window). Implementations should raise
        TrackingError (core.exceptions) on internal failure rather than
        silently resetting all tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracks and reset internal id counters. Used when the
        camera source changes or vision processing restarts, so stale track
        ids from a previous session are never reused."""
