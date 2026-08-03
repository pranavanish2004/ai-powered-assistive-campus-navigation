"""
Heuristic monocular depth estimator.

Design rationale
-----------------
Estimates distance purely from how large a detected object's bounding box
is relative to the frame -- a well-known, decades-old computer vision
heuristic (bigger apparent size ⇒ closer object) that requires no model
weights, no GPU, and no network access. It is deliberately the *default*
`depth_backend` (see `Settings.depth_backend`) for exactly the same reason
`MockObjectDetector` is the default detector: Phase 1.3 must be fully
operable, testable, and demoable without the optional `[vision]` extra.

This is intentionally a coarse approximation -- it assumes a roughly
constant real-world object size and a fixed camera focal length, neither of
which holds precisely across object categories or camera hardware. Distance
*confidence* is set lower than a true depth model would report, and callers
(the obstacle-safety bridge) are expected to treat closer-but-uncertain
estimates conservatively (i.e. still warn) rather than dismiss them for low
confidence -- underestimating danger is worse than a false positive. A real
MiDaS-based estimator (`midas_depth_estimator.py`) can replace this without
any change to the Vision Pipeline, once deployed with real campus footage
to calibrate against.
"""

from __future__ import annotations

import numpy as np

from campus_nav.interfaces.depth_interface import DepthEstimate, DepthEstimatorInterface
from campus_nav.interfaces.detection_interface import BoundingBox

# A rough real-world reference: an object filling this fraction of the
# frame's height is assumed to be at REFERENCE_DISTANCE_METERS. This is a
# single, deliberately simple calibration constant -- accurate calibration
# per object category/camera would be a Phase 2 improvement once real
# campus footage is available to tune against.
_REFERENCE_BBOX_HEIGHT_FRACTION = 0.5
_REFERENCE_DISTANCE_METERS = 2.0

# Estimates are clamped to this range; the heuristic's error grows rapidly
# outside a person-scale-object's plausible near-field distance, so
# reporting an implausible number (e.g. 0.01m or 500m) would be actively
# misleading to a downstream safety decision.
_MIN_DISTANCE_METERS = 0.3
_MAX_DISTANCE_METERS = 15.0

_HEURISTIC_CONFIDENCE = 0.4
"""Deliberately modest and fixed -- see module docstring on why callers
should still act on close, low-confidence estimates rather than discount
them."""


class HeuristicDepthEstimator(DepthEstimatorInterface):
    """Estimates distance from bounding-box height relative to frame
    height. No model weights, no GPU, no network."""

    def __init__(self, frame_height_px: int = 480) -> None:
        self._frame_height_px = frame_height_px

    def estimate_depth(self, frame: np.ndarray, bbox: BoundingBox) -> DepthEstimate:
        frame_height = frame.shape[0] if frame is not None and frame.size else self._frame_height_px
        if frame_height <= 0:
            frame_height = self._frame_height_px

        bbox_height_fraction = bbox.height / frame_height if frame_height > 0 else 0.0
        if bbox_height_fraction <= 0:
            return DepthEstimate(distance_meters=_MAX_DISTANCE_METERS, confidence=0.0)

        # Inverse relationship: an object twice as tall in the frame is
        # assumed to be roughly half the reference distance away.
        distance = _REFERENCE_DISTANCE_METERS * (
            _REFERENCE_BBOX_HEIGHT_FRACTION / bbox_height_fraction
        )
        distance = max(_MIN_DISTANCE_METERS, min(_MAX_DISTANCE_METERS, distance))

        return DepthEstimate(distance_meters=distance, confidence=_HEURISTIC_CONFIDENCE)
