"""
Tests for CameraInterface's contract.

Deliberately does NOT touch OpenCVCamera / real hardware. Instead we define
a minimal FakeCamera to prove the abstraction is usable and enforceable —
this is the whole point of depending on interfaces rather than concrete
classes (see interfaces/camera_interface.py docstring).
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from campus_nav.interfaces.camera_interface import CameraInterface
from campus_nav.interfaces.gps_interface import GPSCoordinate, GPSInterface
from campus_nav.interfaces.speech_recognition_interface import SpeechRecognitionInterface
from campus_nav.interfaces.text_to_speech_interface import (
    SpeechPriority,
    TextToSpeechInterface,
)


class FakeCamera(CameraInterface):
    """Minimal in-memory fake used only for testing code that depends on
    CameraInterface, without needing real camera hardware."""

    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_frame(self) -> np.ndarray | None:
        if not self._started:
            return None
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def is_available(self) -> bool:
        return self._started


def test_cannot_instantiate_abstract_interface_directly():
    with pytest.raises(TypeError):
        CameraInterface()  # type: ignore[abstract]


def test_fake_camera_returns_none_before_start():
    camera = FakeCamera()
    assert camera.get_frame() is None
    assert camera.is_available() is False


def test_fake_camera_context_manager_lifecycle():
    camera = FakeCamera()
    with camera as active_camera:
        assert active_camera.is_available() is True
        frame = active_camera.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    assert camera.is_available() is False


# --- The remaining interfaces follow the identical "cannot instantiate the
# ABC directly, a minimal fake satisfies the contract" pattern. ---


class FakeGPS(GPSInterface):
    def get_current_location(self) -> GPSCoordinate | None:
        return GPSCoordinate(17.59, 78.12, 5.0, datetime.now(timezone.utc))

    def is_available(self) -> bool:
        return True


class FakeSTT(SpeechRecognitionInterface):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return "take me to biomedical"

    def listen_and_transcribe(self, timeout_seconds: float = 5.0) -> str:
        return "take me to biomedical"


class FakeTTS(TextToSpeechInterface):
    def __init__(self) -> None:
        self.last_priority: SpeechPriority | None = None

    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.CONVERSATIONAL) -> None:
        self.last_priority = priority

    def stop(self) -> None:
        self.last_priority = None

    def is_speaking(self) -> bool:
        return False


def test_cannot_instantiate_gps_stt_tts_interfaces_directly():
    with pytest.raises(TypeError):
        GPSInterface()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        SpeechRecognitionInterface()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        TextToSpeechInterface()  # type: ignore[abstract]


def test_fake_gps_returns_coordinate():
    gps = FakeGPS()
    location = gps.get_current_location()
    assert location is not None
    assert gps.is_available() is True


def test_fake_stt_transcribes():
    stt = FakeSTT()
    assert stt.listen_and_transcribe() == "take me to biomedical"


def test_fake_tts_records_priority():
    tts = FakeTTS()
    tts.speak("Obstacle ahead, stop.", priority=SpeechPriority.SAFETY)
    assert tts.last_priority is SpeechPriority.SAFETY
