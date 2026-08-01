"""OpenCV VideoCapture adapter implementing CameraInterface."""

from __future__ import annotations

import cv2
import numpy as np

from campus_nav.core.exceptions import CameraError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.camera_interface import CameraInterface

logger = get_logger(__name__)


class OpenCVCamera(CameraInterface):
    """Camera adapter backed by cv2.VideoCapture — suitable for a laptop
    webcam during development. A phone-camera or embedded-camera adapter can
    implement the same CameraInterface later with no change to callers."""

    def __init__(self, device_index: int, frame_width: int, frame_height: int) -> None:
        self._device_index = device_index
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return  # idempotent

        capture = cv2.VideoCapture(self._device_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        if not capture.isOpened():
            raise CameraError(f"Could not open camera at device index {self._device_index}.")

        self._capture = capture
        logger.info("Camera started", extra={"device_index": self._device_index})

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Camera stopped", extra={"device_index": self._device_index})

    def get_frame(self) -> np.ndarray | None:
        if self._capture is None or not self._capture.isOpened():
            return None

        success, frame = self._capture.read()
        if not success:
            logger.warning("Failed to read frame from camera; returning None")
            return None
        return frame

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.isOpened()
