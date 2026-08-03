"""
OCR (optical character recognition) abstraction.

Used to read building names and signboards from camera frames. Kept behind
an interface for the same reason as every other vision backend: the
concrete engine (EasyOCR, PaddleOCR, a future alternative) is a heavy,
swappable implementation detail; the pipeline and its tests should not care
which one is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from campus_nav.interfaces.detection_interface import BoundingBox


@dataclass(frozen=True, slots=True)
class OCRResult:
    """A single recognized text region."""

    text: str
    confidence: float
    bbox: BoundingBox


class OCRInterface(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def read_text(self, frame: np.ndarray, roi: BoundingBox | None = None) -> list[OCRResult]:
        """Extract text from `frame`, optionally restricted to a region of
        interest (`roi`) -- e.g. the bounding box of a detected sign or
        door, so OCR isn't wasted scanning an entire frame when only a
        specific detected object is a plausible text source. Implementations
        should filter out results below their configured confidence
        threshold themselves (see OCRInterface implementations' docstrings)
        rather than returning everything and expecting the caller to filter."""
