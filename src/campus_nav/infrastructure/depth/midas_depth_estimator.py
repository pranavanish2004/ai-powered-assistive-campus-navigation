"""
MiDaS-backed monocular depth estimator.

IMPORTANT -- optional dependency
--------------------------------
Requires `torch` (see pyproject.toml's `[vision]` extra). The MiDaS model
itself is downloaded on first use via `torch.hub`, which additionally
requires network access the first time it runs -- unlike every other
backend in this codebase, this one is not fully offline until its weights
are cached locally. Install with:

    pip install -e ".[vision]"
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campus_nav.core.exceptions import DepthEstimationError, ModelLoadError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.depth_interface import DepthEstimate, DepthEstimatorInterface
from campus_nav.interfaces.detection_interface import BoundingBox

logger = get_logger(__name__)

_INSTALL_HINT = (
    "torch is not installed. Install the optional vision extra with: " 'pip install -e ".[vision]"'
)

# MiDaS produces a *relative* inverse-depth map, not calibrated meters. This
# scale factor is a rough, uncalibrated conversion -- proper calibration
# against known campus distances is a Phase 2 deployment task once the
# model is actually run against real campus footage.
_UNCALIBRATED_SCALE_METERS = 10.0
_MIDAS_CONFIDENCE = 0.6


class MiDaSDepthEstimator(DepthEstimatorInterface):
    """Depth estimation backed by Intel's pretrained MiDaS model via
    torch.hub (no fine-tuning, per Phase 1 scope)."""

    def __init__(self, model_type: str = "MiDaS_small") -> None:
        self._model_type = model_type
        self._model: Any | None = None
        self._transform: Any | None = None

    def _get_model(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                import torch  # local import: optional, heavy dependency
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                logger.info("Loading MiDaS depth model", extra={"model_type": self._model_type})
                model = torch.hub.load("intel-isl/MiDaS", self._model_type)
                model.eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                transform = (
                    transforms.small_transform
                    if "small" in self._model_type.lower()
                    else transforms.default_transform
                )
                self._model = model
                self._transform = transform
            except Exception as exc:
                raise ModelLoadError(f"Failed to load MiDaS model: {exc}") from exc

        return self._model, self._transform

    def estimate_depth(self, frame: np.ndarray, bbox: BoundingBox) -> DepthEstimate:
        import torch  # safe: _get_model() already required torch to be importable

        model, transform = self._get_model()

        try:
            input_batch = transform(frame)
            with torch.no_grad():
                prediction = model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()
        except Exception as exc:
            raise DepthEstimationError(f"MiDaS inference failed: {exc}") from exc

        region = depth_map[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]
        if region.size == 0:
            raise DepthEstimationError("Bounding box produced an empty depth region.")

        # MiDaS outputs inverse depth (higher value = closer); invert and
        # apply the uncalibrated scale to approximate meters.
        mean_inverse_depth = float(region.mean())
        if mean_inverse_depth <= 0:
            raise DepthEstimationError("MiDaS produced a non-positive depth value.")

        distance_meters = _UNCALIBRATED_SCALE_METERS / mean_inverse_depth
        return DepthEstimate(distance_meters=distance_meters, confidence=_MIDAS_CONFIDENCE)
