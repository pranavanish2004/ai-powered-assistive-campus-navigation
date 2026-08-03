"""Tests for detection interfaces, BoundingBox geometry, category mapping,
and MockObjectDetector."""

import numpy as np
import pytest

from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector, make_detection
from campus_nav.interfaces.detection_interface import BoundingBox, ObjectDetectorInterface
from campus_nav.vision.detection.categories import (
    ObjectCategory,
    category_for_label,
    is_always_announce,
    is_conditional,
)

pytestmark = pytest.mark.unit


def test_bounding_box_dimensions():
    bbox = BoundingBox(10, 20, 50, 80)
    assert bbox.width == 40
    assert bbox.height == 60
    assert bbox.area == 2400
    assert bbox.center_x == 30
    assert bbox.center_y == 50


def test_bounding_box_negative_dimensions_clamped_to_zero():
    bbox = BoundingBox(50, 50, 10, 10)  # inverted coordinates
    assert bbox.width == 0
    assert bbox.height == 0
    assert bbox.area == 0


def test_iou_identical_boxes_is_one():
    bbox = BoundingBox(0, 0, 100, 100)
    assert bbox.iou(bbox) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(100, 100, 110, 110)
    assert a.iou(b) == 0.0


def test_iou_partial_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    # Intersection: 5x5=25, union: 100+100-25=175
    assert a.iou(b) == pytest.approx(25 / 175)


def test_iou_zero_area_boxes_returns_zero():
    a = BoundingBox(0, 0, 0, 0)
    b = BoundingBox(0, 0, 0, 0)
    assert a.iou(b) == 0.0


def test_cannot_instantiate_detector_interface_directly():
    with pytest.raises(TypeError):
        ObjectDetectorInterface()  # type: ignore[abstract]


def test_mock_detector_returns_empty_by_default():
    detector = MockObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_mock_detector_returns_configured_detections():
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50))
    detector = MockObjectDetector([detection])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result == [detection]


def test_mock_detector_ignores_frame_content():
    detector = MockObjectDetector(
        [make_detection(ObjectCategory.DOOR, "door", 0.8, (0, 0, 10, 10))]
    )
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    assert detector.detect(frame1) == detector.detect(frame2)


def test_mock_detector_set_detections_updates_subsequent_calls():
    detector = MockObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []

    new_detection = make_detection(ObjectCategory.PERSON, "person", 0.95, (0, 0, 20, 40))
    detector.set_detections([new_detection])
    assert detector.detect(frame) == [new_detection]


@pytest.mark.parametrize(
    "label,expected_category",
    [
        ("person", ObjectCategory.PERSON),
        ("PERSON", ObjectCategory.PERSON),
        ("  car  ", ObjectCategory.VEHICLE),
        ("bench", ObjectCategory.BENCH),
        ("dog", ObjectCategory.ANIMAL),
        ("some_never_seen_label", ObjectCategory.UNKNOWN),
    ],
)
def test_category_for_label(label, expected_category):
    assert category_for_label(label) is expected_category


def test_always_announce_categories():
    assert is_always_announce(ObjectCategory.DOOR)
    assert is_always_announce(ObjectCategory.STAIRS)
    assert is_always_announce(ObjectCategory.RAMP)
    assert not is_always_announce(ObjectCategory.BENCH)


def test_conditional_categories_include_unknown():
    assert is_conditional(ObjectCategory.BENCH)
    assert is_conditional(ObjectCategory.UNKNOWN)
    assert not is_conditional(ObjectCategory.DOOR)


def test_detection_is_immutable():
    detection = make_detection(ObjectCategory.TREE, "potted plant", 0.7, (0, 0, 5, 5))
    with pytest.raises(AttributeError):
        detection.confidence = 0.9  # type: ignore[misc]
