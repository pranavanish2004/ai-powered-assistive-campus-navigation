"""
CameraManager -- frame acquisition pipeline with buffering and async capture.

Design rationale
-----------------
Vision processing (detection/OCR/depth) is far slower than frame capture.
If the API/pipeline pulled frames synchronously on every request, a slow
processing step would starve frame capture and the "live" feed would fall
further and further behind. Instead, `CameraManager` runs a dedicated
background thread that continuously captures frames into a small, bounded
ring buffer (`collections.deque(maxlen=...)`), and `get_latest_frame()`
always returns the most recent frame available -- never a queued backlog.

This deliberately favors *recency* over *completeness*: for a safety-
adjacent obstacle-warning system, a slightly-stale frame is actively
misleading ("the bench is right there" when it's actually five seconds and
several meters of walking behind you), so dropping old frames rather than
processing every single one is the correct trade-off, not an oversight.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from campus_nav.core.exceptions import CameraError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.camera_interface import CameraInterface

logger = get_logger(__name__)


@dataclass
class CameraManagerDiagnostics:
    """Snapshot of CameraManager's operational health -- surfaced via the
    `/vision/camera/status` API endpoint."""

    is_running: bool = False
    frames_captured: int = 0
    frames_dropped: int = 0
    """Frames overwritten in the ring buffer before being consumed -- an
    expected, healthy occurrence at steady state (see module docstring),
    not itself a sign of a problem unless it dominates frames_captured."""
    last_capture_error: str | None = None
    actual_fps: float = 0.0


class CameraManager:
    """Wraps a `CameraInterface` with a background capture thread and a
    small bounded frame buffer."""

    def __init__(
        self, camera: CameraInterface, buffer_size: int = 5, target_fps: float = 15.0
    ) -> None:
        self._camera = camera
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._target_interval_seconds = 1.0 / target_fps if target_fps > 0 else 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._frames_captured = 0
        self._frames_dropped = 0
        self._last_capture_error: str | None = None
        self._fps_window_start = 0.0
        self._fps_window_count = 0
        self._actual_fps = 0.0

    def start(self) -> None:
        """Start the camera and begin capturing frames on a background
        thread. Idempotent: calling start() while already running is a
        no-op, matching CameraInterface.start()'s own idempotency contract."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._camera.start()
        self._stop_event.clear()
        self._fps_window_start = time.monotonic()
        self._fps_window_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("CameraManager started")

    def stop(self) -> None:
        """Stop the capture thread and release the underlying camera.
        Safe to call even if start() was never called."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._camera.stop()
        logger.info("CameraManager stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame, or None if nothing has
        been captured yet."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_diagnostics(self) -> CameraManagerDiagnostics:
        with self._lock:
            return CameraManagerDiagnostics(
                is_running=self.is_running(),
                frames_captured=self._frames_captured,
                frames_dropped=self._frames_dropped,
                last_capture_error=self._last_capture_error,
                actual_fps=self._actual_fps,
            )

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                frame = self._camera.get_frame()
            except CameraError as exc:
                with self._lock:
                    self._last_capture_error = str(exc)
                logger.warning("Camera capture error", extra={"error": str(exc)})
                frame = None

            if frame is not None:
                with self._lock:
                    was_full = len(self._buffer) == self._buffer.maxlen
                    self._buffer.append(frame)
                    self._frames_captured += 1
                    if was_full:
                        self._frames_dropped += 1
                    self._last_capture_error = None
                    self._update_fps_locked()

            elapsed = time.monotonic() - loop_start
            sleep_time = self._target_interval_seconds - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def _update_fps_locked(self) -> None:
        """Must be called while holding self._lock."""
        self._fps_window_count += 1
        now = time.monotonic()
        window_duration = now - self._fps_window_start
        if window_duration >= 1.0:
            self._actual_fps = self._fps_window_count / window_duration
            self._fps_window_start = now
            self._fps_window_count = 0
