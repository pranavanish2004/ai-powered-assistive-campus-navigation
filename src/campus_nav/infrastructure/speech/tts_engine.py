"""
pyttsx3-backed text-to-speech implementation.

pyttsx3 runs fully offline (wraps the OS's native TTS engine) — a deliberate
choice tied to the safety-lane architecture: obstacle warnings must never
depend on network availability. Cloud TTS (higher quality voices) can be
added later as a second implementation for the conversational lane only,
selected via Settings.tts_engine — the safety lane should always be wired
to an offline engine like this one.
"""

from __future__ import annotations

import threading
from typing import Any

from campus_nav.core.exceptions import TextToSpeechError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.text_to_speech_interface import SpeechPriority, TextToSpeechInterface

logger = get_logger(__name__)


class Pyttsx3TTS(TextToSpeechInterface):
    """Offline TTS engine. Speech synthesis is guarded by a lock so a
    SAFETY-priority utterance can interrupt CONVERSATIONAL speech that is
    still playing, per the interface contract."""

    def __init__(self, rate_wpm: int = 175) -> None:
        self._rate_wpm = rate_wpm
        self._lock = threading.Lock()
        self._speaking = False
        self._engine: Any | None = None  # lazy-loaded: touches audio hardware on init

    def _get_engine(self) -> Any:
        if self._engine is None:
            import pyttsx3  # local import: audio hardware dependency

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate_wpm)
                self._engine = engine
            except Exception as exc:  # pragma: no cover - hardware-dependent
                raise TextToSpeechError(f"Failed to initialize TTS engine: {exc}") from exc
        return self._engine

    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.CONVERSATIONAL) -> None:
        engine = self._get_engine()

        with self._lock:
            if priority is SpeechPriority.SAFETY and self._speaking:
                logger.info("SAFETY utterance preempting in-progress speech")
                engine.stop()

            self._speaking = True
            logger.debug("Speaking", extra={"priority": priority.name, "text": text})
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                self._speaking = False

    def stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking
