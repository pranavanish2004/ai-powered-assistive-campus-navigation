"""
Speech-to-text abstraction.

Whisper is the Phase-1 choice for the *implementation* (added as an optional
dependency once the speech module is built), but the Conversation Manager
and Intent Detection modules depend on this interface only. Swapping in a
lower-latency streaming STT engine later is a new adapter, not a refactor of
every caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SpeechRecognitionInterface(ABC):
    """Abstract base class for speech-to-text engines."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a chunk of mono PCM audio into text. Implementations
        should raise SpeechRecognitionError (core.exceptions) if the audio
        cannot be transcribed."""

    @abstractmethod
    def listen_and_transcribe(self, timeout_seconds: float = 5.0) -> str:
        """Capture audio from the default microphone for up to
        `timeout_seconds` and return the transcribed text."""
