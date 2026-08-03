"""Tests for vision.pipeline.vision_factory -- Settings-driven backend
selection for detector/OCR/depth/tracker, and the mock defaults that keep
the app bootable without the optional [vision] extra installed."""

import pytest

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector
from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.vision.pipeline.vision_factory import (
    create_depth_estimator,
    create_detector,
    create_landmark_detector,
    create_ocr_service,
    create_tracker,
    create_vision_pipeline,
)
from campus_nav.vision.pipeline.vision_pipeline import VisionPipeline
from campus_nav.vision.tracking.tracker import IOUTracker

pytestmark = pytest.mark.unit


def test_default_settings_select_mock_detector():
    settings = Settings(_env_file=None)
    assert isinstance(create_detector(settings), MockObjectDetector)


def test_default_settings_select_mock_ocr():
    settings = Settings(_env_file=None)
    assert isinstance(create_ocr_service(settings), MockOCRService)


def test_default_settings_select_heuristic_depth_estimator():
    settings = Settings(_env_file=None)
    assert isinstance(create_depth_estimator(settings), HeuristicDepthEstimator)


def test_default_settings_select_null_landmark_detector():
    settings = Settings(_env_file=None)
    assert isinstance(create_landmark_detector(settings), NullLandmarkDetector)


def test_tracker_uses_configured_thresholds():
    settings = Settings(_env_file=None, tracking_iou_threshold=0.5, tracking_max_missed_frames=7)
    tracker = create_tracker(settings)
    assert isinstance(tracker, IOUTracker)
    assert tracker._iou_threshold == 0.5
    assert tracker._max_missed_frames == 7


def test_create_vision_pipeline_returns_fully_wired_pipeline():
    settings = Settings(_env_file=None)
    pipeline = create_vision_pipeline(settings)
    assert isinstance(pipeline, VisionPipeline)


def test_yolov8_backend_selected_returns_yolo_detector_instance():
    from campus_nav.infrastructure.detection.yolo_detector import YOLOv8Detector

    settings = Settings(_env_file=None, detector_backend="yolov8", yolo_model_path="yolov8n.pt")
    detector = create_detector(settings)
    assert isinstance(detector, YOLOv8Detector)
    # Not calling .detect() here -- that would require the optional
    # [vision] extra (ultralytics) to be installed; see test_detection.py
    # / infrastructure adapter docstrings for the lazy-import contract.


def test_easyocr_backend_selected_returns_easyocr_instance():
    from campus_nav.infrastructure.ocr.easyocr_service import EasyOCRService

    settings = Settings(_env_file=None, ocr_backend="easyocr")
    ocr = create_ocr_service(settings)
    assert isinstance(ocr, EasyOCRService)


def test_midas_backend_selected_returns_midas_instance():
    from campus_nav.infrastructure.depth.midas_depth_estimator import MiDaSDepthEstimator

    settings = Settings(_env_file=None, depth_backend="midas")
    estimator = create_depth_estimator(settings)
    assert isinstance(estimator, MiDaSDepthEstimator)
