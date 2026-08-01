"""Tests for NullTTS -- the safety-priority interruption contract must hold
even for the no-op implementation used in tests/API, since callers (e.g.
VoiceGuidance) rely on the SpeechPriority being recorded faithfully."""

import pytest

from campus_nav.infrastructure.speech.null_tts import NullTTS
from campus_nav.interfaces.text_to_speech_interface import SpeechPriority

pytestmark = pytest.mark.unit


def test_records_spoken_utterances():
    tts = NullTTS()
    tts.speak("hello")
    assert tts.last_utterance is not None
    assert tts.last_utterance.text == "hello"
    assert tts.last_utterance.priority is SpeechPriority.CONVERSATIONAL


def test_records_safety_priority():
    tts = NullTTS()
    tts.speak("Obstacle ahead", priority=SpeechPriority.SAFETY)
    assert tts.last_utterance.priority is SpeechPriority.SAFETY


def test_history_accumulates_in_order():
    tts = NullTTS()
    tts.speak("first")
    tts.speak("second")
    texts = [u.text for u in tts.history]
    assert texts == ["first", "second"]


def test_last_utterance_is_none_when_nothing_spoken():
    tts = NullTTS()
    assert tts.last_utterance is None


def test_is_speaking_false_after_speak_returns():
    tts = NullTTS()
    tts.speak("hello")
    assert tts.is_speaking() is False


def test_stop_resets_speaking_flag():
    tts = NullTTS()
    tts.speak("hello")
    tts.stop()
    assert tts.is_speaking() is False
