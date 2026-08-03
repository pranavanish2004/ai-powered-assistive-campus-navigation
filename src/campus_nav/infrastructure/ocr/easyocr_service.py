"""
EasyOCR-backed OCR adapter.

IMPORTANT -- optional dependency
--------------------------------
`easyocr` is NOT part of the default install (see pyproject.toml's
`[vision]` extra) -- it pulls in `torch`. Install with:

    pip install -e ".[vision]"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campus_nav.core.exceptions import ModelLoadError, OCRError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.detection_interface import BoundingBox
from campus_nav.interfaces.ocr_interface import OCRInterface, OCRResult

logger = get_logger(__name__)

_INSTALL_HINT = (
    "easyocr is not installed. Install the optional vision extra with: "
    'pip install -e ".[vision]"'
)


class EasyOCRService(OCRInterface):
    """OCR backed by EasyOCR, restricted to English by default (sufficient
    for the IIT-H pilot's building signage) and offline after its model
    weights are first downloaded."""

    def __init__(self, confidence_threshold: float, languages: list[str] | None = None) -> None:
        self._confidence_threshold = confidence_threshold
        self._languages = languages or ["en"]
        self._reader: Any | None = None  # lazy-loaded on first use

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr  # local import: optional, heavy dependency
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                logger.info("Loading EasyOCR reader", extra={"languages": self._languages})
                self._reader = easyocr.Reader(self._languages, gpu=False)
            except Exception as exc:
                raise ModelLoadError(f"Failed to initialize EasyOCR reader: {exc}") from exc
        return self._reader

    def read_text(self, frame: np.ndarray, roi: BoundingBox | None = None) -> list[OCRResult]:
        reader = self._get_reader()

        region = frame
        offset_x, offset_y = 0, 0
        if roi is not None:
            region = frame[roi.y_min : roi.y_max, roi.x_min : roi.x_max]
            offset_x, offset_y = roi.x_min, roi.y_min

        try:
            raw_results = reader.readtext(region)
        except Exception as exc:
            raise OCRError(f"EasyOCR inference failed: {exc}") from exc

        results: list[OCRResult] = []
        for polygon, text, confidence in raw_results:
            if confidence < self._confidence_threshold:
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            bbox = BoundingBox(
                x_min=int(min(xs)) + offset_x,
                y_min=int(min(ys)) + offset_y,
                x_max=int(max(xs)) + offset_x,
                y_max=int(max(ys)) + offset_y,
            )
            results.append(OCRResult(text=text, confidence=float(confidence), bbox=bbox))

        return results
