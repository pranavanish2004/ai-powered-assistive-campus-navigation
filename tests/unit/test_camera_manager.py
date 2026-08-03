"""Tests for CameraManager's async capture loop and bounded frame buffer."""

import time

import numpy as np
import pytest

from campus_nav.interfaces.camera_interface import CameraInterface
from campus_nav.vision.camera.camera_manager import CameraManager

pytestmark = pytest.mark.unit


class FakeCamera(CameraInterface):
    """A camera that returns a fresh frame with an incrementing marker on
    every call, so tests can distinguish 'the latest frame' from stale
    ones without timing assumptions about real camera hardware."""

    def __init__(self) -> None:
        self._started = False
        self._counter = 0

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_frame(self):
        if not self._started:
            return None
        self._counter += 1
        frame = np.full((4, 4, 3), self._counter, dtype=np.uint8)
        return frame

    def is_available(self) -> bool:
        return self._started


def test_start_and_stop_lifecycle():
    manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=50.0)
    assert manager.is_running() is False

    manager.start()
    assert manager.is_running() is True

    manager.stop()
    assert manager.is_running() is False


def test_start_is_idempotent():
    manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=50.0)
    manager.start()
    manager.start()  # should not raise or spawn a second thread
    assert manager.is_running() is True
    manager.stop()


def test_stop_before_start_is_safe():
    manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=50.0)
    manager.stop()  # should not raise


def test_get_latest_frame_returns_none_before_start():
    manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=50.0)
    assert manager.get_latest_frame() is None


def test_frames_are_captured_after_start():
    manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=50.0)
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.get_latest_frame() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        frame = manager.get_latest_frame()
        assert frame is not None
    finally:
        manager.stop()


def test_diagnostics_report_frames_captured():
    manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=50.0)
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.get_diagnostics().frames_captured == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        diagnostics = manager.get_diagnostics()
        assert diagnostics.frames_captured > 0
        assert diagnostics.is_running is True
    finally:
        manager.stop()


def test_buffer_drops_old_frames_beyond_capacity():
    manager = CameraManager(FakeCamera(), buffer_size=2, target_fps=200.0)
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.get_diagnostics().frames_captured < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        diagnostics = manager.get_diagnostics()
        assert diagnostics.frames_dropped > 0
    finally:
        manager.stop()


def test_capture_error_is_recorded_in_diagnostics():
    from campus_nav.core.exceptions import CameraError

    class FailingCamera(CameraInterface):
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def get_frame(self):
            raise CameraError("simulated capture failure")

        def is_available(self) -> bool:
            return True

    manager = CameraManager(FailingCamera(), buffer_size=5, target_fps=50.0)
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.get_diagnostics().last_capture_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        diagnostics = manager.get_diagnostics()
        assert diagnostics.last_capture_error is not None
    finally:
        manager.stop()
