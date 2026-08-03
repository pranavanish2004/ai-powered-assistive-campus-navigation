"""
Video file camera adapter -- replays a video file as if it were a live
camera. Implements the same `CameraInterface` as `OpenCVCamera`.

Primary use: deterministic development, demos, and integration tests
against realistic frame sequences without needing physical camera hardware
or a live network stream -- important given this project's stated goal of
a "hardware-independent, testable architecture."
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from campus_nav.core.exceptions import CameraError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.camera_interface import CameraInterface

logger = get_logger(__name__)


class VideoFileCamera(CameraInterface):
    """Camera adapter backed by a local video file."""

    def __init__(self, file_path: str | Path, loop: bool = True) -> None:
        self._file_path = str(file_path)
        self._loop = loop
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return  # idempotent

        capture = cv2.VideoCapture(self._file_path)
        if not capture.isOpened():
            raise CameraError(f"Could not open video file at {self._file_path!r}.")

        self._capture = capture
        logger.info("Video file camera started", extra={"file_path": self._file_path})

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Video file camera stopped", extra={"file_path": self._file_path})

    def get_frame(self) -> np.ndarray | None:
        if self._capture is None or not self._capture.isOpened():
            return None

        success, frame = self._capture.read()
        if not success:
            if self._loop:
                # Reached end of file: rewind and try once more, so a video
                # file behaves like a continuous live source for long-
                # running demos/tests rather than going permanently dark.
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self._capture.read()
            if not success:
                logger.warning("Failed to read frame from video file; returning None")
                return None
        return frame

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.isOpened()
