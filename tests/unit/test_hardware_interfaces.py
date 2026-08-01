"""Tests for the Camera/GPS/STT/TTS interface contracts, using minimal fakes
-- no hardware required, per the interfaces' design rationale."""

from datetime import UTC, datetime

import numpy as np
import pytest

from campus_nav.interfaces.camera_interface import CameraInterface
from campus_nav.interfaces.gps_interface import GPSCoordinate, GPSInterface
from campus_nav.interfaces.speech_recognition_interface import SpeechRecognitionInterface
from campus_nav.interfaces.text_to_speech_interface import SpeechPriority, TextToSpeechInterface

pytestmark = pytest.mark.unit


class FakeCamera(CameraInterface):
    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8) if self._started else None

    def is_available(self) -> bool:
        return self._started


class FakeGPS(GPSInterface):
    def get_current_location(self):
        return GPSCoordinate(17.59, 78.12, 5.0, datetime.now(UTC))

    def is_available(self) -> bool:
        return True


class FakeSTT(SpeechRecognitionInterface):
    def transcribe(self, audio, sample_rate) -> str:
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


def test_cannot_instantiate_abstract_interfaces_directly():
    for interface_class in (
        CameraInterface,
        GPSInterface,
        SpeechRecognitionInterface,
        TextToSpeechInterface,
    ):
        with pytest.raises(TypeError):
            interface_class()  # type: ignore[abstract]


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
