"""
Mock object detector.

Returns a pre-configured, deterministic set of detections regardless of
frame content. This is the default `detector_backend` (see
`Settings.detector_backend`) so the application boots and the vision
pipeline is fully exercisable without the optional `[vision]` extra
installed or any model weights downloaded -- exactly the role
`MockGPSProvider` plays for Phase 1.2 localization.
"""

from __future__ import annotations

import numpy as np

from campus_nav.interfaces.detection_interface import (
    BoundingBox,
    Detection,
    ObjectDetectorInterface,
)
from campus_nav.vision.detection.categories import ObjectCategory


class MockObjectDetector(ObjectDetectorInterface):
    """Returns a fixed, caller-configurable list of detections on every
    call to `detect()`, ignoring actual frame content. Useful both as the
    safe default runtime backend and as a fully controllable test double."""

    def __init__(self, fixed_detections: list[Detection] | None = None) -> None:
        self._fixed_detections = fixed_detections if fixed_detections is not None else []

    def set_detections(self, detections: list[Detection]) -> None:
        """Test/dev helper to change what subsequent detect() calls return,
        simulating a different frame's contents without a real camera."""
        self._fixed_detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._fixed_detections)


def make_detection(
    category: ObjectCategory,
    label: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
) -> Detection:
    """Convenience constructor for tests and demos: build a Detection from
    a plain (x_min, y_min, x_max, y_max) tuple instead of a BoundingBox."""
    x_min, y_min, x_max, y_max = bbox
    return Detection(
        category=category,
        label=label,
        confidence=confidence,
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
    )
