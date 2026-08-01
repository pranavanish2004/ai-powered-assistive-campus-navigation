"""
Text-to-speech abstraction.

This is the boundary where the safety/conversational latency split becomes
concrete: `speak()` takes an explicit `priority` so a safety alert
("Obstacle ahead, stop.") can interrupt an in-progress conversational
sentence. Without this parameter in the interface from day one, safety
alerts would queue up behind chatty MLLM output — exactly the coupling this
architecture is designed against. See docs/architecture.md, Section 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto


class SpeechPriority(Enum):
    """Determines whether new speech should interrupt currently playing
    speech. SAFETY speech must always be able to preempt CONVERSATIONAL
    speech; the reverse must never happen."""

    CONVERSATIONAL = auto()
    SAFETY = auto()


class TextToSpeechInterface(ABC):
    """Abstract base class for text-to-speech engines."""

    @abstractmethod
    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.CONVERSATIONAL) -> None:
        """Speak `text` aloud. If `priority` is SAFETY, implementations MUST
        interrupt any currently playing CONVERSATIONAL speech and speak
        immediately. A SAFETY utterance must never be dropped or queued
        behind conversational output."""

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop any currently playing speech."""

    @abstractmethod
    def is_speaking(self) -> bool:
        """Return True if the engine is currently producing audio."""
