"""
Monocular depth estimation abstraction.

Estimates how far away a detected object is, from a single camera frame
(no stereo rig, no LiDAR -- consistent with Phase 1's "pretrained models
only, no new hardware" constraint). Used to compute a safe-distance
judgment for obstacles: a bench 10 meters away is irrelevant; the same
bench 0.5 meters away, directly ahead, is worth an alert.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from campus_nav.interfaces.detection_interface import BoundingBox


@dataclass(frozen=True, slots=True)
class DepthEstimate:
    """Estimated distance to a specific detected object."""

    distance_meters: float
    confidence: float
    """In [0.0, 1.0]. Monocular depth estimation is inherently approximate;
    callers (the Safety Engine bridge) should treat low-confidence
    estimates conservatively -- e.g. still warn if a low-confidence
    estimate is close, since underestimating danger is worse than a false
    positive."""


class DepthEstimatorInterface(ABC):
    """Abstract base class for monocular depth estimation backends."""

    @abstractmethod
    def estimate_depth(self, frame: np.ndarray, bbox: BoundingBox) -> DepthEstimate:
        """Estimate the distance to the object occupying `bbox` within
        `frame`. Implementations should raise DepthEstimationError
        (core.exceptions) on failure rather than returning a fabricated
        confident estimate."""
