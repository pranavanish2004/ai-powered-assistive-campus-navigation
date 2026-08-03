"""Integration tests for VisionService -- CameraManager + VisionPipeline
running together on a real background thread, using a FakeCamera and mock
vision backends (no physical hardware, no ML dependencies required)."""

import time

import numpy as np
import pytest

from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector, make_detection
from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.interfaces.camera_interface import CameraInterface
from campus_nav.vision.camera.camera_manager import CameraManager
from campus_nav.vision.detection.categories import ObjectCategory
from campus_nav.vision.pipeline.vision_pipeline import VisionPipeline
from campus_nav.vision.pipeline.vision_service import VisionService
from campus_nav.vision.tracking.tracker import IOUTracker

pytestmark = pytest.mark.integration


class FakeCamera(CameraInterface):
    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8) if self._started else None

    def is_available(self) -> bool:
        return self._started


def _wait_until(predicate, timeout_seconds=2.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_service(detections=None, target_fps=30.0):
    detector = MockObjectDetector(detections or [])
    pipeline = VisionPipeline(
        detector=detector,
        tracker=IOUTracker(),
        depth_estimator=HeuristicDepthEstimator(480),
        ocr=MockOCRService(),
        landmark_detector=NullLandmarkDetector(),
    )
    camera_manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=30.0)
    return VisionService(
        camera_manager=camera_manager,
        pipeline=pipeline,
        safe_distance_meters=1.5,
        target_fps=target_fps,
    )


def test_start_and_stop_lifecycle():
    service = _make_service()
    assert service.is_running() is False
    service.start()
    assert service.is_running() is True
    service.stop()
    assert service.is_running() is False


def test_processes_frames_and_produces_results():
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (280, 200, 360, 470))
    service = _make_service(detections=[detection])
    service.start()
    try:
        assert _wait_until(lambda: service.get_latest_result() is not None)
        result = service.get_latest_result()
        assert len(result.tracked_objects) == 1
    finally:
        service.stop()


def test_obstacles_are_derived_from_processed_frames():
    detection = make_detection(ObjectCategory.STAIRS, "stairs", 0.9, (0, 200, 40, 400))
    service = _make_service(detections=[detection])
    service.start()
    try:
        assert _wait_until(lambda: len(service.get_latest_obstacles()) > 0)
        obstacles = service.get_latest_obstacles()
        assert obstacles[0].announce_text == "Stairs ahead."
    finally:
        service.stop()


def test_diagnostics_report_processed_frame_count():
    service = _make_service()
    service.start()
    try:
        assert _wait_until(lambda: service.get_diagnostics().frames_processed > 0)
        diagnostics = service.get_diagnostics()
        assert diagnostics.is_running is True
        assert diagnostics.camera.is_running is True
    finally:
        service.stop()


def test_stop_before_start_is_safe():
    service = _make_service()
    service.stop()  # should not raise


def test_no_result_before_any_processing():
    service = _make_service()
    assert service.get_latest_result() is None
    assert service.get_latest_obstacles() == []
