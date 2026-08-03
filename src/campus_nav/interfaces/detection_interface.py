"""
Object detection abstraction.

Design rationale
-----------------
Mirrors the Phase 1.1/1.2 pattern: the Vision Pipeline depends only on
`ObjectDetectorInterface`, never on `ultralytics`/YOLO directly. This keeps
the pipeline testable with `MockObjectDetector` (no model weights, no GPU,
no network) and swappable to a different detection backend later without
touching pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from campus_nav.vision.detection.categories import ObjectCategory


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space axis-aligned bounding box. (0, 0) is the top-left of the
    frame, matching OpenCV's convention."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union with another box, in [0.0, 1.0]. Used by
        the tracker to match detections across frames."""
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        intersection = inter_width * inter_height

        union = self.area + other.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detected in one frame, before tracking is applied."""

    category: ObjectCategory
    label: str
    """Raw model class name (e.g. 'person', 'chair') — kept alongside the
    normalized ObjectCategory so callers can log/debug against the model's
    actual vocabulary."""
    confidence: float
    bbox: BoundingBox


class ObjectDetectorInterface(ABC):
    """Abstract base class for object detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame (OpenCV convention) and
        return all detections at or above the configured confidence
        threshold. Implementations should raise DetectionError
        (core.exceptions) on inference failure rather than returning a
        misleadingly empty list."""
