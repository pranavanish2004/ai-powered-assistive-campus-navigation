"""
Voice guidance.

The single place where NavigationInstruction/safety-alert text becomes an
actual spoken utterance. Keeping this thin and separate from
NavigationEngine matters for testability: engine logic (route progress,
state transitions) can be unit-tested with a fake TTS that just records
calls, without pulling in pyttsx3 or any audio hardware.
"""

from __future__ import annotations

from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.navigation_interface import NavigationInstruction
from campus_nav.interfaces.text_to_speech_interface import SpeechPriority, TextToSpeechInterface

logger = get_logger(__name__)


class VoiceGuidance:
    """Speaks navigation instructions and safety alerts through a
    TextToSpeechInterface, applying the correct SpeechPriority to each."""

    def __init__(self, tts: TextToSpeechInterface) -> None:
        self._tts = tts

    def announce_instruction(self, instruction: NavigationInstruction) -> None:
        """Turn-by-turn guidance is conversational-lane: it can be
        interrupted by a safety alert, but does not itself preempt other
        speech."""
        logger.debug("Announcing instruction", extra={"text": instruction.text})
        self._tts.speak(instruction.text, priority=SpeechPriority.CONVERSATIONAL)

    def announce_safety_alert(self, message: str) -> None:
        """Safety alerts must interrupt anything currently playing. This is
        the only path through which NavigationEngine/SafetyEngine output
        should reach the user for time-critical warnings."""
        logger.info("Announcing safety alert", extra={"text": message})
        self._tts.speak(message, priority=SpeechPriority.SAFETY)
