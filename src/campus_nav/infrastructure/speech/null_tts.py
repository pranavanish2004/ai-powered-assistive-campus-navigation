"""
No-op TTS implementation.

The FastAPI server process has no speaker attached to it, and unit/
integration tests must not touch real audio hardware (pyttsx3.init() will
fail or hang in a headless CI container). This implementation satisfies
TextToSpeechInterface by recording what *would* have been spoken instead of
producing audio -- callers (the API layer) surface that text in the HTTP
response body instead, and tests assert against the recorded calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from campus_nav.interfaces.text_to_speech_interface import SpeechPriority, TextToSpeechInterface


@dataclass(frozen=True, slots=True)
class SpokenUtterance:
    text: str
    priority: SpeechPriority


class NullTTS(TextToSpeechInterface):
    """Records spoken utterances instead of producing audio."""

    def __init__(self) -> None:
        self._history: list[SpokenUtterance] = []
        self._speaking = False

    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.CONVERSATIONAL) -> None:
        self._history.append(SpokenUtterance(text=text, priority=priority))
        self._speaking = False  # synthesis is instantaneous for this no-op engine

    def stop(self) -> None:
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def history(self) -> list[SpokenUtterance]:
        return list(self._history)

    @property
    def last_utterance(self) -> SpokenUtterance | None:
        return self._history[-1] if self._history else None
