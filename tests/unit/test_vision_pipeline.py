"""Integration-style tests for VisionPipeline: composing real (mock)
detection, tracking, depth, OCR, and landmark components together."""

import numpy as np
import pytest

from campus_nav.core.exceptions import DetectionError, OCRError
from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector, make_detection
from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.vision.detection.categories import ObjectCategory
from campus_nav.vision.pipeline.vision_pipeline import VisionPipeline
from campus_nav.vision.tracking.tracker import IOUTracker

pytestmark = pytest.mark.unit


def _make_pipeline(detector=None, tracker=None, depth=None, ocr=None, landmark=None):
    return VisionPipeline(
        detector=detector or MockObjectDetector(),
        tracker=tracker or IOUTracker(),
        depth_estimator=depth or HeuristicDepthEstimator(480),
        ocr=ocr or MockOCRService(),
        landmark_detector=landmark or NullLandmarkDetector(),
    )


def test_empty_frame_produces_empty_result():
    pipeline = _make_pipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)

    assert result.tracked_objects == []
    assert result.depth_estimates == {}
    assert result.ocr_results == []
    assert result.landmark_matches == []
    assert result.frame_width == 640
    assert result.frame_height == 480


def test_detection_flows_through_to_tracked_objects_with_depth():
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (280, 200, 360, 470))
    pipeline = _make_pipeline(detector=MockObjectDetector([detection]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)

    assert len(result.tracked_objects) == 1
    track_id = result.tracked_objects[0].track_id
    assert track_id in result.depth_estimates
    assert result.depth_estimates[track_id].distance_meters > 0


def test_multiple_frames_maintain_persistent_track_ids():
    detection = make_detection(ObjectCategory.PERSON, "person", 0.9, (100, 100, 150, 300))
    detector = MockObjectDetector([detection])
    pipeline = _make_pipeline(detector=detector)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result1 = pipeline.process_frame(frame)
    result2 = pipeline.process_frame(frame)

    assert result1.tracked_objects[0].track_id == result2.tracked_objects[0].track_id
    assert result2.tracked_objects[0].frames_tracked == 2


def test_detection_failure_is_isolated_and_pipeline_continues():
    class FailingDetector(MockObjectDetector):
        def detect(self, frame):
            raise DetectionError("simulated failure")

    pipeline = _make_pipeline(detector=FailingDetector())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)  # must not raise
    assert result.tracked_objects == []


def test_ocr_failure_is_isolated_and_pipeline_continues():
    class FailingOCR(MockOCRService):
        def read_text(self, frame, roi=None):
            raise OCRError("simulated OCR failure")

    detection = make_detection(ObjectCategory.DOOR, "door", 0.9, (0, 0, 50, 50))
    pipeline = _make_pipeline(detector=MockObjectDetector([detection]), ocr=FailingOCR())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.process_frame(frame)  # must not raise
    assert result.ocr_results == []
    assert len(result.tracked_objects) == 1  # unaffected by OCR failure


def test_processing_time_is_recorded_and_non_negative():
    pipeline = _make_pipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame)
    assert result.processing_time_seconds >= 0.0
