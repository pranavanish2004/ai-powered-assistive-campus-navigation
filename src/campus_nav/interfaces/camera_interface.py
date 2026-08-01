"""
Camera abstraction.

The Vision Pipeline (Phase 1.3) must depend only on this interface, never on
OpenCV or a specific camera SDK directly:

- Testability: unit tests inject a fake camera with canned frames, no
  physical webcam required (important for CI).
- Swappability: moving from a laptop webcam (Phase 1 dev) to a chest-mounted
  camera or phone stream (real deployment) is a new adapter class, not a
  vision-pipeline rewrite.
- Fail-safe integration: `is_available()` lets the Decision Engine check
  camera health *before* trusting vision-based guidance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

import numpy as np


class CameraInterface(ABC):
    """Abstract base class for any camera/video frame source."""

    @abstractmethod
    def start(self) -> None:
        """Open/initialize the underlying device. Must be idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Release the underlying device. Safe to call even if start()
        was never called."""

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame as a BGR numpy array, or None if no
        frame is currently available. Returns None instead of raising on a
        transient miss — callers decide whether a single missed frame
        matters, rather than every camera hiccup becoming an exception."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the camera is currently opened and healthy."""

    def __enter__(self) -> CameraInterface:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()
