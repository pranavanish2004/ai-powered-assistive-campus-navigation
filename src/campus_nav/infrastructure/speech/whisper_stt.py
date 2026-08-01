"""
OpenAI Whisper adapter implementing SpeechRecognitionInterface.

IMPORTANT — optional dependency
--------------------------------
`openai-whisper` (and `sounddevice`) are NOT part of the default install.
They pull in heavy ML dependencies (torch, numba) not needed to develop or
test Phase 1.1/1.2, and are deferred to when the Speech Recognition module
is actually built out. Install them with:

    pip install -e ".[speech]"

This module is safe to import even without that extra installed — the
`whisper`/`sounddevice` imports are deferred to inside the methods that
need them, and raise a clear ConfigurationError (not an opaque ImportError)
if the extra hasn't been installed.
"""

from __future__ import annotations

import numpy as np

from campus_nav.core.exceptions import ConfigurationError, SpeechRecognitionError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.speech_recognition_interface import SpeechRecognitionInterface

logger = get_logger(__name__)

_INSTALL_HINT = (
    "openai-whisper is not installed. Install the optional speech extra with: "
    'pip install -e ".[speech]"'
)


class WhisperSTT(SpeechRecognitionInterface):
    """Speech-to-text backed by OpenAI's Whisper (pretrained, no fine-tuning
    per Phase 1 scope)."""

    def __init__(self, model_size: str = "base", language: str = "en") -> None:
        self._model_size = model_size
        self._language = language
        self._model: object | None = None  # lazy-loaded on first use

    def _get_model(self) -> object:
        if self._model is None:
            try:
                import whisper  # local import: optional, heavy dependency
            except ImportError as exc:
                raise ConfigurationError(_INSTALL_HINT) from exc

            logger.info("Loading Whisper model", extra={"model_size": self._model_size})
            self._model = whisper.load_model(self._model_size)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        model = self._get_model()
        audio_float32 = audio.astype(np.float32)

        result = model.transcribe(audio_float32, language=self._language)  # type: ignore[attr-defined]
        text = str(result.get("text", "")).strip()

        if not text:
            raise SpeechRecognitionError("Whisper returned empty transcription.")

        logger.debug("Transcribed audio", extra={"text": text})
        return text

    def listen_and_transcribe(self, timeout_seconds: float = 5.0) -> str:
        try:
            import sounddevice as sd  # local import: optional, audio hardware dependency
        except ImportError as exc:
            raise ConfigurationError(_INSTALL_HINT) from exc

        sample_rate = 16000
        logger.debug("Recording microphone", extra={"timeout_seconds": timeout_seconds})
        recording = sd.rec(
            int(timeout_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = recording.flatten()
        return self.transcribe(audio, sample_rate)
