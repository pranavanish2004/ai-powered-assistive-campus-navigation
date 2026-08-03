"""
YOLOv8 object detector adapter.

IMPORTANT — optional dependency
--------------------------------
`ultralytics` (and its `torch` dependency) is NOT part of the default
install — see pyproject.toml's `[vision]` extra. This mirrors the
`WhisperSTT` pattern from Phase 1.1: the import is deferred to inside the
method that needs it, and raises a clear `ModelLoadError` (not an opaque
`ImportError`) if the extra hasn't been installed. Install with:

    pip install -e ".[vision]"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campus_nav.core.exceptions import DetectionError, ModelLoadError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.detection_interface import (
    BoundingBox,
    Detection,
    ObjectDetectorInterface,
)
from campus_nav.vision.detection.categories import category_for_label

logger = get_logger(__name__)

_INSTALL_HINT = (
    "ultralytics is not installed. Install the optional vision extra with: "
    'pip install -e ".[vision]"'
)


class YOLOv8Detector(ObjectDetectorInterface):
    """Object detection backed by a pretrained YOLOv8 model (no
    fine-tuning, per Phase 1 scope — see original spec's Phase 2 section
    for when custom training becomes in-scope)."""

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._model: Any | None = None  # lazy-loaded on first use

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO  # local import: optional, heavy dependency
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                logger.info("Loading YOLOv8 model", extra={"model_path": self._model_path})
                self._model = YOLO(self._model_path)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load YOLOv8 weights from {self._model_path!r}: {exc}"
                ) from exc
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._get_model()

        try:
            results = model.predict(frame, conf=self._confidence_threshold, verbose=False)
        except Exception as exc:
            raise DetectionError(f"YOLOv8 inference failed: {exc}") from exc

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {})

            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self._confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                label = str(names.get(class_id, "unknown"))
                x_min, y_min, x_max, y_max = (int(v) for v in box.xyxy[0].tolist())

                detections.append(
                    Detection(
                        category=category_for_label(label),
                        label=label,
                        confidence=confidence,
                        bbox=BoundingBox(x_min, y_min, x_max, y_max),
                    )
                )

        return detections
