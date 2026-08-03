"""Tests for vision.safety.obstacle_events -- the object detection policy
(ignore beside-path, warn on blocking/always-relevant categories)."""

from datetime import UTC, datetime

import pytest

from campus_nav.interfaces.depth_interface import DepthEstimate
from campus_nav.interfaces.detection_interface import BoundingBox
from campus_nav.interfaces.obstacle_interface import ObstaclePosition
from campus_nav.interfaces.tracking_interface import TrackedObject
from campus_nav.vision.detection.categories import ObjectCategory
from campus_nav.vision.pipeline.frame_result import VisionFrameResult
from campus_nav.vision.safety.obstacle_events import build_obstacle_events, classify_position

pytestmark = pytest.mark.unit


def _tracked(category, bbox, track_id=1, confidence=0.9):
    return TrackedObject(
        track_id=track_id,
        category=category,
        label=category.value,
        confidence=confidence,
        bbox=bbox,
        frames_tracked=5,
        frames_missed=0,
    )


def _frame_result(tracked_objects, depth_estimates=None, frame_width=640, frame_height=480):
    return VisionFrameResult(
        timestamp=datetime.now(UTC),
        tracked_objects=tracked_objects,
        depth_estimates=depth_estimates or {},
        ocr_results=[],
        landmark_matches=[],
        frame_width=frame_width,
        frame_height=frame_height,
        processing_time_seconds=0.01,
    )


# --- classify_position ---


def test_centered_bbox_is_blocking():
    bbox = BoundingBox(300, 200, 340, 400)  # centered in a 640-wide frame
    assert classify_position(bbox, frame_width=640) is ObstaclePosition.BLOCKING


def test_edge_bbox_is_beside_path():
    bbox = BoundingBox(0, 200, 40, 400)  # far left edge
    assert classify_position(bbox, frame_width=640) is ObstaclePosition.BESIDE_PATH


def test_zero_frame_width_defaults_to_beside_path():
    bbox = BoundingBox(0, 0, 10, 10)
    assert classify_position(bbox, frame_width=0) is ObstaclePosition.BESIDE_PATH


# --- build_obstacle_events: always-announce categories ---


@pytest.mark.parametrize(
    "category,expected_text",
    [
        (ObjectCategory.DOOR, "The entrance is directly ahead."),
        (ObjectCategory.RAMP, "Ramp ahead."),
        (ObjectCategory.STAIRS, "Stairs ahead."),
    ],
)
def test_always_announce_categories_speak_regardless_of_position(category, expected_text):
    # Off to the side (beside path) -- should still announce.
    bbox = BoundingBox(0, 200, 40, 400)
    tracked = _tracked(category, bbox)
    result = _frame_result([tracked])

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text == expected_text


# --- build_obstacle_events: conditional categories (tree/bench/pole) ---


def test_bench_beside_path_is_silent():
    bbox = BoundingBox(0, 200, 40, 400)  # beside path
    tracked = _tracked(ObjectCategory.BENCH, bbox)
    result = _frame_result([tracked], depth_estimates={1: DepthEstimate(1.0, 0.8)})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text is None
    assert events[0].position is ObstaclePosition.BESIDE_PATH


def test_bench_blocking_and_close_triggers_announcement():
    bbox = BoundingBox(300, 200, 340, 400)  # centered = blocking
    tracked = _tracked(ObjectCategory.BENCH, bbox)
    result = _frame_result([tracked], depth_estimates={1: DepthEstimate(1.0, 0.8)})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text == "Obstacle ahead."


def test_bench_blocking_but_far_away_is_silent():
    bbox = BoundingBox(300, 200, 340, 400)  # centered = blocking
    tracked = _tracked(ObjectCategory.BENCH, bbox)
    result = _frame_result([tracked], depth_estimates={1: DepthEstimate(10.0, 0.8)})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text is None


def test_bench_blocking_with_no_depth_estimate_still_warns():
    # No depth estimate available -- policy treats this conservatively
    # (still warn) rather than assuming it's safely far away.
    bbox = BoundingBox(300, 200, 340, 400)
    tracked = _tracked(ObjectCategory.BENCH, bbox)
    result = _frame_result([tracked], depth_estimates={})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text == "Obstacle ahead."


# --- build_obstacle_events: person handling ---


def test_person_close_triggers_someone_is_approaching():
    bbox = BoundingBox(0, 200, 40, 400)  # even off to the side
    tracked = _tracked(ObjectCategory.PERSON, bbox)
    result = _frame_result([tracked], depth_estimates={1: DepthEstimate(1.0, 0.8)})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text == "Someone is approaching."


def test_person_far_away_is_silent():
    bbox = BoundingBox(300, 200, 340, 400)
    tracked = _tracked(ObjectCategory.PERSON, bbox)
    result = _frame_result([tracked], depth_estimates={1: DepthEstimate(10.0, 0.8)})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text is None


# --- unknown category treated as conditional ---


def test_unknown_category_blocking_and_close_warns():
    bbox = BoundingBox(300, 200, 340, 400)
    tracked = _tracked(ObjectCategory.UNKNOWN, bbox)
    result = _frame_result([tracked], depth_estimates={1: DepthEstimate(0.5, 0.5)})

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events[0].announce_text == "Obstacle ahead."


# --- one event per tracked object, even silent ones ---


def test_one_event_produced_per_tracked_object():
    bbox = BoundingBox(0, 200, 40, 400)
    tracked_objects = [
        _tracked(ObjectCategory.BENCH, bbox, track_id=1),
        _tracked(ObjectCategory.TREE, bbox, track_id=2),
    ]
    result = _frame_result(tracked_objects)

    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert len(events) == 2
    assert {e.track_id for e in events} == {1, 2}


def test_empty_tracked_objects_produces_no_events():
    result = _frame_result([])
    events = build_obstacle_events(result, safe_distance_meters=1.5)
    assert events == []
