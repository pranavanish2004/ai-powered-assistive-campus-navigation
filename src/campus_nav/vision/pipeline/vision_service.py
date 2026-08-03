"""
VisionService -- ties `CameraManager` (async frame acquisition) and
`VisionPipeline` (detection/tracking/depth/OCR/landmarks) together into a
single background processing loop, and is the one object the API layer
holds a reference to (analogous to how `api/dependencies.py` holds the
Phase 1.2 navigation engine registry).

Why a separate processing loop from capture
---------------------------------------------
`CameraManager` already runs its own capture thread at the camera's native
rate (see its module docstring). Vision processing (YOLO inference, OCR,
depth estimation) is far more expensive per-frame than capture, so it runs
on its *own* thread at `Settings.vision_processing_fps`, always pulling
whatever the *latest* buffered frame is -- naturally skipping frames when
processing can't keep up, rather than building a backlog.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.obstacle_interface import ObstacleEvent
from campus_nav.vision.camera.camera_manager import CameraManager, CameraManagerDiagnostics
from campus_nav.vision.pipeline.frame_result import VisionFrameResult
from campus_nav.vision.pipeline.vision_pipeline import VisionPipeline
from campus_nav.vision.safety.obstacle_events import build_obstacle_events

logger = get_logger(__name__)


@dataclass
class VisionServiceDiagnostics:
    """Snapshot of the full vision pipeline's operational health --
    surfaced via the `/vision/status` and `/vision/diagnostics` endpoints."""

    is_running: bool
    frames_processed: int
    last_process_error: str | None
    actual_processing_fps: float
    last_result_timestamp: str | None
    camera: CameraManagerDiagnostics


class VisionService:
    """Background-threaded orchestrator: pulls frames from a
    `CameraManager` and runs them through a `VisionPipeline` at a
    configurable target rate."""

    def __init__(
        self,
        camera_manager: CameraManager,
        pipeline: VisionPipeline,
        safe_distance_meters: float,
        target_fps: float = 5.0,
    ) -> None:
        self._camera_manager = camera_manager
        self._pipeline = pipeline
        self._safe_distance_meters = safe_distance_meters
        self._target_interval_seconds = 1.0 / target_fps if target_fps > 0 else 0.0

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._latest_result: VisionFrameResult | None = None
        self._latest_obstacles: list[ObstacleEvent] = []
        self._frames_processed = 0
        self._last_process_error: str | None = None
        self._fps_window_start = 0.0
        self._fps_window_count = 0
        self._actual_fps = 0.0

    def start(self) -> None:
        """Start camera capture and begin processing frames. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._camera_manager.start()
        self._stop_event.clear()
        self._fps_window_start = time.monotonic()
        self._fps_window_count = 0
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        logger.info("VisionService started")

    def stop(self) -> None:
        """Stop processing and release the camera. Safe to call even if
        start() was never called."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._camera_manager.stop()
        logger.info("VisionService stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_latest_result(self) -> VisionFrameResult | None:
        with self._lock:
            return self._latest_result

    def get_latest_obstacles(self) -> list[ObstacleEvent]:
        with self._lock:
            return list(self._latest_obstacles)

    def get_diagnostics(self) -> VisionServiceDiagnostics:
        with self._lock:
            return VisionServiceDiagnostics(
                is_running=self.is_running(),
                frames_processed=self._frames_processed,
                last_process_error=self._last_process_error,
                actual_processing_fps=self._actual_fps,
                last_result_timestamp=(
                    self._latest_result.timestamp.isoformat() if self._latest_result else None
                ),
                camera=self._camera_manager.get_diagnostics(),
            )

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            frame = self._camera_manager.get_latest_frame()

            if frame is not None:
                try:
                    result = self._pipeline.process_frame(frame)
                    obstacles = build_obstacle_events(result, self._safe_distance_meters)
                    with self._lock:
                        self._latest_result = result
                        self._latest_obstacles = obstacles
                        self._frames_processed += 1
                        self._last_process_error = None
                        self._update_fps_locked()
                except Exception as exc:  # noqa: BLE001 - isolate pipeline failures per frame
                    with self._lock:
                        self._last_process_error = str(exc)
                    logger.warning("Vision pipeline processing error", extra={"error": str(exc)})

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
