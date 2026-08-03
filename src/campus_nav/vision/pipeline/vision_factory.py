"""
Vision component factory.

Centralizes backend selection (mock vs. real) for detection, OCR, and
depth estimation, driven by `Settings`. This is where the "safe default
that needs no [vision] extra" policy is enforced in one place -- swapping
`detector_backend` from 'mock' to 'yolov8' in configuration is the entire
change needed to go from the always-available default to the real model,
with zero changes to `VisionPipeline` or any caller.
"""

from __future__ import annotations

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.infrastructure.depth.midas_depth_estimator import MiDaSDepthEstimator
from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector
from campus_nav.infrastructure.detection.yolo_detector import YOLOv8Detector
from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.infrastructure.ocr.easyocr_service import EasyOCRService
from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.interfaces.depth_interface import DepthEstimatorInterface
from campus_nav.interfaces.detection_interface import ObjectDetectorInterface
from campus_nav.interfaces.landmark_interface import LandmarkDetectorInterface
from campus_nav.interfaces.ocr_interface import OCRInterface
from campus_nav.interfaces.tracking_interface import ObjectTrackerInterface
from campus_nav.vision.pipeline.vision_pipeline import VisionPipeline
from campus_nav.vision.tracking.tracker import IOUTracker


def create_detector(settings: Settings) -> ObjectDetectorInterface:
    if settings.detector_backend == "yolov8":
        return YOLOv8Detector(
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.min_detection_confidence,
        )
    return MockObjectDetector()


def create_ocr_service(settings: Settings) -> OCRInterface:
    if settings.ocr_backend == "easyocr":
        return EasyOCRService(confidence_threshold=settings.ocr_confidence_threshold)
    return MockOCRService()


def create_depth_estimator(settings: Settings) -> DepthEstimatorInterface:
    if settings.depth_backend == "midas":
        return MiDaSDepthEstimator()
    return HeuristicDepthEstimator(frame_height_px=settings.camera_frame_height)


def create_tracker(settings: Settings) -> ObjectTrackerInterface:
    return IOUTracker(
        iou_threshold=settings.tracking_iou_threshold,
        max_missed_frames=settings.tracking_max_missed_frames,
    )


def create_landmark_detector(settings: Settings) -> LandmarkDetectorInterface:
    return NullLandmarkDetector()


def create_vision_pipeline(settings: Settings) -> VisionPipeline:
    """Build a fully-wired VisionPipeline using whichever backends
    `settings` selects."""
    return VisionPipeline(
        detector=create_detector(settings),
        tracker=create_tracker(settings),
        depth_estimator=create_depth_estimator(settings),
        ocr=create_ocr_service(settings),
        landmark_detector=create_landmark_detector(settings),
    )
