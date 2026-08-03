"""
VisionPipeline -- the Phase 1.3 orchestrator.

Composes the five vision components (detection, tracking, depth
estimation, OCR, landmark detection) into a single `process_frame()` call,
mirroring the role `NavigationEngine` plays for Phase 1.2's routing/
localization/safety/instruction components: this class contains no
detection math, no tracking math, and no OCR/depth code of its own -- it
only sequences calls to the modules that do, which is what keeps each of
those modules independently unit-testable.

Failure isolation
-------------------
Each stage's failure is caught and logged individually rather than
aborting the whole frame: a depth estimation failure for one object
shouldn't discard that frame's otherwise-successful detections, and an OCR
failure shouldn't prevent obstacle tracking from continuing. This matches
the Phase 1.1 exception-hierarchy philosophy -- these are all
`RecoverableError` subclasses precisely because losing one signal for one
frame must not take down the whole pipeline.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np

from campus_nav.core.exceptions import DepthEstimationError, DetectionError, OCRError, TrackingError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.depth_interface import DepthEstimate, DepthEstimatorInterface
from campus_nav.interfaces.detection_interface import ObjectDetectorInterface
from campus_nav.interfaces.landmark_interface import LandmarkDetectorInterface
from campus_nav.interfaces.ocr_interface import OCRInterface
from campus_nav.interfaces.tracking_interface import ObjectTrackerInterface
from campus_nav.vision.pipeline.frame_result import VisionFrameResult

logger = get_logger(__name__)


class VisionPipeline:
    """Runs one frame through detection, tracking, depth estimation, OCR,
    and landmark detection, producing a single `VisionFrameResult`."""

    def __init__(
        self,
        detector: ObjectDetectorInterface,
        tracker: ObjectTrackerInterface,
        depth_estimator: DepthEstimatorInterface,
        ocr: OCRInterface,
        landmark_detector: LandmarkDetectorInterface,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._depth_estimator = depth_estimator
        self._ocr = ocr
        self._landmark_detector = landmark_detector

    def process_frame(self, frame: np.ndarray) -> VisionFrameResult:
        start_time = time.monotonic()

        try:
            detections = self._detector.detect(frame)
        except DetectionError as exc:
            logger.warning(
                "Detection failed for frame; treating as no detections", extra={"error": str(exc)}
            )
            detections = []

        try:
            tracked_objects = self._tracker.update(detections)
        except TrackingError as exc:
            logger.warning(
                "Tracking failed for frame; resetting tracker", extra={"error": str(exc)}
            )
            self._tracker.reset()
            tracked_objects = []

        depth_estimates: dict[int, DepthEstimate] = {}
        for tracked in tracked_objects:
            try:
                depth_estimates[tracked.track_id] = self._depth_estimator.estimate_depth(
                    frame, tracked.bbox
                )
            except DepthEstimationError as exc:
                logger.debug(
                    "Depth estimation failed for tracked object; omitting distance",
                    extra={"track_id": tracked.track_id, "error": str(exc)},
                )

        try:
            ocr_results = self._ocr.read_text(frame)
        except OCRError as exc:
            logger.debug("OCR failed for frame; omitting text results", extra={"error": str(exc)})
            ocr_results = []

        landmark_matches = self._landmark_detector.detect_landmarks(frame)

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        processing_time = time.monotonic() - start_time

        return VisionFrameResult(
            timestamp=datetime.now(UTC),
            tracked_objects=tracked_objects,
            depth_estimates=depth_estimates,
            ocr_results=ocr_results,
            landmark_matches=landmark_matches,
            frame_width=frame_width,
            frame_height=frame_height,
            processing_time_seconds=processing_time,
        )
