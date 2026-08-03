"""Tests for OCR interface and MockOCRService."""

import numpy as np
import pytest

from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.interfaces.detection_interface import BoundingBox
from campus_nav.interfaces.ocr_interface import OCRInterface, OCRResult

pytestmark = pytest.mark.unit


def test_cannot_instantiate_ocr_interface_directly():
    with pytest.raises(TypeError):
        OCRInterface()  # type: ignore[abstract]


def test_mock_ocr_returns_empty_by_default():
    ocr = MockOCRService()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert ocr.read_text(frame) == []


def test_mock_ocr_returns_all_results_without_roi():
    result = OCRResult(text="Library", confidence=0.9, bbox=BoundingBox(10, 10, 100, 40))
    ocr = MockOCRService([result])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert ocr.read_text(frame) == [result]


def test_mock_ocr_filters_by_roi_overlap():
    inside = OCRResult(text="Biomedical", confidence=0.85, bbox=BoundingBox(0, 0, 50, 50))
    outside = OCRResult(text="Canteen", confidence=0.7, bbox=BoundingBox(500, 500, 550, 550))
    ocr = MockOCRService([inside, outside])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    roi = BoundingBox(0, 0, 60, 60)
    result = ocr.read_text(frame, roi=roi)
    assert result == [inside]


def test_mock_ocr_roi_with_no_overlap_returns_empty():
    result = OCRResult(text="Library", confidence=0.9, bbox=BoundingBox(10, 10, 50, 50))
    ocr = MockOCRService([result])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    roi = BoundingBox(1000, 1000, 1050, 1050)
    assert ocr.read_text(frame, roi=roi) == []


def test_set_results_updates_subsequent_calls():
    ocr = MockOCRService()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert ocr.read_text(frame) == []

    result = OCRResult(text="SNCC", confidence=0.8, bbox=BoundingBox(0, 0, 30, 30))
    ocr.set_results([result])
    assert ocr.read_text(frame) == [result]
