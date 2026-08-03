"""Mock OCR service -- deterministic, ML-free default (see mock_detector.py
for the identical rationale applied to OCR)."""

from __future__ import annotations

import numpy as np

from campus_nav.interfaces.detection_interface import BoundingBox
from campus_nav.interfaces.ocr_interface import OCRInterface, OCRResult


class MockOCRService(OCRInterface):
    """Returns a fixed, caller-configurable list of text results,
    optionally filtered to only those overlapping a requested ROI."""

    def __init__(self, fixed_results: list[OCRResult] | None = None) -> None:
        self._fixed_results = fixed_results if fixed_results is not None else []

    def set_results(self, results: list[OCRResult]) -> None:
        self._fixed_results = results

    def read_text(self, frame: np.ndarray, roi: BoundingBox | None = None) -> list[OCRResult]:
        if roi is None:
            return list(self._fixed_results)
        return [r for r in self._fixed_results if r.bbox.iou(roi) > 0.0]
