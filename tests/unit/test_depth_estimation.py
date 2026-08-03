"""Tests for the depth estimation interface and HeuristicDepthEstimator."""

import numpy as np
import pytest

from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.interfaces.depth_interface import DepthEstimatorInterface
from campus_nav.interfaces.detection_interface import BoundingBox

pytestmark = pytest.mark.unit


def test_cannot_instantiate_depth_interface_directly():
    with pytest.raises(TypeError):
        DepthEstimatorInterface()  # type: ignore[abstract]


def test_larger_bbox_estimated_closer_than_smaller_bbox():
    estimator = HeuristicDepthEstimator(frame_height_px=480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    small_bbox = BoundingBox(0, 0, 50, 50)
    large_bbox = BoundingBox(0, 0, 50, 400)

    small_estimate = estimator.estimate_depth(frame, small_bbox)
    large_estimate = estimator.estimate_depth(frame, large_bbox)

    assert large_estimate.distance_meters < small_estimate.distance_meters


def test_distance_is_clamped_to_reasonable_range():
    estimator = HeuristicDepthEstimator(frame_height_px=480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tiny_bbox = BoundingBox(0, 0, 1, 1)
    huge_bbox = BoundingBox(0, 0, 640, 480)

    tiny_estimate = estimator.estimate_depth(frame, tiny_bbox)
    huge_estimate = estimator.estimate_depth(frame, huge_bbox)

    assert 0.3 <= tiny_estimate.distance_meters <= 15.0
    assert 0.3 <= huge_estimate.distance_meters <= 15.0


def test_zero_height_bbox_returns_max_distance_with_zero_confidence():
    estimator = HeuristicDepthEstimator(frame_height_px=480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    zero_height_bbox = BoundingBox(0, 100, 50, 100)  # y_min == y_max
    estimate = estimator.estimate_depth(frame, zero_height_bbox)

    assert estimate.confidence == 0.0
    assert estimate.distance_meters == 15.0


def test_confidence_is_fixed_and_moderate():
    estimator = HeuristicDepthEstimator(frame_height_px=480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(0, 0, 50, 200)

    estimate = estimator.estimate_depth(frame, bbox)
    assert 0.0 < estimate.confidence < 1.0


def test_falls_back_to_configured_height_for_empty_frame():
    estimator = HeuristicDepthEstimator(frame_height_px=480)
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    bbox = BoundingBox(0, 0, 50, 100)

    # Should not raise, and should use the configured fallback height.
    estimate = estimator.estimate_depth(empty_frame, bbox)
    assert estimate.distance_meters > 0
