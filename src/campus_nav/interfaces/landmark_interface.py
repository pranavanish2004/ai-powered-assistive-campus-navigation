"""
Landmark detection abstraction.

Phase 1.3 scope, per the project specification, is the *interface* for a
future landmark recognition model -- actual landmark recognition (matching
a camera frame against trained campus-specific visual landmarks) is a
Phase 2 deliverable that requires collected campus imagery and evaluation,
which the specification explicitly defers. The default implementation
shipped in Phase 1.3 (`NullLandmarkDetector`) always returns no matches; it
exists so the Vision Pipeline has a concrete, wireable dependency today,
and so a real model can be substituted later with zero pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from campus_nav.interfaces.detection_interface import BoundingBox


@dataclass(frozen=True, slots=True)
class LandmarkMatch:
    """A recognized campus landmark within a frame."""

    node_id: str
    """The campus graph Node.id this landmark corresponds to (see
    navigation.graph.models.Node.landmark_description) -- this is the
    integration point that will eventually let vision-based landmark
    recognition contribute to localization confidence, alongside GPS."""
    confidence: float
    bbox: BoundingBox


class LandmarkDetectorInterface(ABC):
    """Abstract base class for campus landmark recognition backends."""

    @abstractmethod
    def detect_landmarks(self, frame: np.ndarray) -> list[LandmarkMatch]:
        """Return any recognized campus landmarks in `frame`. An empty list
        is a completely normal result (most frames contain no recognizable
        landmark) -- this is not an error condition."""
