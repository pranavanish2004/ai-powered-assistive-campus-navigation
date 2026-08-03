"""
IP camera adapter -- connects to a network camera stream (e.g. DroidCam,
IP Webcam on Android) via its HTTP(S)/RTSP stream URL.

Implements the same `CameraInterface` defined in Phase 1.1
(`interfaces/camera_interface.py`) that `OpenCVCamera` implements -- no
changes to that interface were needed. This is exactly the extensibility
the interface's docstring called out: "moving from a laptop webcam to a
phone camera stream is a new adapter class, not a rewrite."
"""

from __future__ import annotations

import cv2
import numpy as np

from campus_nav.core.exceptions import CameraError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.camera_interface import CameraInterface

logger = get_logger(__name__)


class IPCameraSource(CameraInterface):
    """Camera adapter backed by a network stream URL (DroidCam, IP Webcam,
    RTSP, or any other source OpenCV's VideoCapture can open by URL)."""

    def __init__(self, stream_url: str) -> None:
        if not stream_url:
            raise CameraError("IPCameraSource requires a non-empty stream_url.")
        self._stream_url = stream_url
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return  # idempotent

        capture = cv2.VideoCapture(self._stream_url)
        if not capture.isOpened():
            raise CameraError(f"Could not open IP camera stream at {self._stream_url!r}.")

        self._capture = capture
        logger.info("IP camera stream started", extra={"stream_url": self._stream_url})

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("IP camera stream stopped", extra={"stream_url": self._stream_url})

    def get_frame(self) -> np.ndarray | None:
        if self._capture is None or not self._capture.isOpened():
            return None

        success, frame = self._capture.read()
        if not success:
            logger.warning("Failed to read frame from IP camera stream; returning None")
            return None
        return frame

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.isOpened()
