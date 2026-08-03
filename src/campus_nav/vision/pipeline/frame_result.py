"""Aggregate result of processing a single frame through the full vision
pipeline (detection -> tracking -> depth -> OCR -> landmarks)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from campus_nav.interfaces.depth_interface import DepthEstimate
from campus_nav.interfaces.landmark_interface import LandmarkMatch
from campus_nav.interfaces.ocr_interface import OCRResult
from campus_nav.interfaces.tracking_interface import TrackedObject


@dataclass(frozen=True, slots=True)
class VisionFrameResult:
    """Everything the Vision Pipeline produced for one processed frame."""

    timestamp: datetime
    tracked_objects: list[TrackedObject]
    depth_estimates: dict[int, DepthEstimate]
    """Keyed by TrackedObject.track_id. A track_id absent from this dict
    means depth estimation was unavailable or failed for that object this
    frame -- not that the object is at distance zero."""
    ocr_results: list[OCRResult]
    landmark_matches: list[LandmarkMatch]
    frame_width: int
    frame_height: int
    processing_time_seconds: float
