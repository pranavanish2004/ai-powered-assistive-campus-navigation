"""
Placeholder landmark detector.

Always returns no matches. This is the Phase 1.3 default and only
implementation -- per the project specification, training/evaluating a
real campus landmark recognition model is explicitly Phase 2 scope
(requires collected campus imagery, dataset curation, and evaluation
against pretrained baselines first). Wiring this interface into the
Vision Pipeline now means a real model becomes a one-class swap later,
with zero pipeline changes.
"""

from __future__ import annotations

import numpy as np

from campus_nav.interfaces.landmark_interface import LandmarkDetectorInterface, LandmarkMatch


class NullLandmarkDetector(LandmarkDetectorInterface):
    """Always returns an empty list of landmark matches."""

    def detect_landmarks(self, frame: np.ndarray) -> list[LandmarkMatch]:
        return []
