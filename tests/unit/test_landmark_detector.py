"""Tests for the landmark detection interface and NullLandmarkDetector."""

import numpy as np
import pytest

from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.interfaces.landmark_interface import LandmarkDetectorInterface, LandmarkMatch

pytestmark = pytest.mark.unit


def test_cannot_instantiate_landmark_interface_directly():
    with pytest.raises(TypeError):
        LandmarkDetectorInterface()  # type: ignore[abstract]


def test_null_landmark_detector_always_returns_empty_list():
    detector = NullLandmarkDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect_landmarks(frame) == []


def test_null_landmark_detector_handles_arbitrary_frame_content():
    detector = NullLandmarkDetector()
    random_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert detector.detect_landmarks(random_frame) == []


def test_landmark_match_is_immutable():
    from campus_nav.interfaces.detection_interface import BoundingBox

    match = LandmarkMatch(node_id="library", confidence=0.9, bbox=BoundingBox(0, 0, 10, 10))
    with pytest.raises(AttributeError):
        match.confidence = 0.5  # type: ignore[misc]
