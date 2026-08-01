"""Tests for VoiceGuidance -- verifies the correct SpeechPriority is applied
for instructions (CONVERSATIONAL) vs safety alerts (SAFETY), which is the
whole point of this class existing as a thin, testable layer."""

import pytest

from campus_nav.infrastructure.speech.null_tts import NullTTS
from campus_nav.interfaces.navigation_interface import NavigationInstruction
from campus_nav.interfaces.text_to_speech_interface import SpeechPriority
from campus_nav.navigation.engine.voice_guidance import VoiceGuidance

pytestmark = pytest.mark.unit


def test_announce_instruction_uses_conversational_priority():
    tts = NullTTS()
    voice = VoiceGuidance(tts)
    instruction = NavigationInstruction(
        text="Continue for 50 meters.", target_node_id="B", distance_meters=50.0
    )
    voice.announce_instruction(instruction)
    assert tts.last_utterance.priority is SpeechPriority.CONVERSATIONAL
    assert tts.last_utterance.text == "Continue for 50 meters."


def test_announce_safety_alert_uses_safety_priority():
    tts = NullTTS()
    voice = VoiceGuidance(tts)
    voice.announce_safety_alert("You have moved off the route.")
    assert tts.last_utterance.priority is SpeechPriority.SAFETY
    assert tts.last_utterance.text == "You have moved off the route."


def test_safety_alert_after_instruction_is_recorded_separately():
    tts = NullTTS()
    voice = VoiceGuidance(tts)
    instruction = NavigationInstruction(text="Turn left.", target_node_id="B", distance_meters=10.0)
    voice.announce_instruction(instruction)
    voice.announce_safety_alert("Obstacle ahead.")
    assert [u.priority for u in tts.history] == [
        SpeechPriority.CONVERSATIONAL,
        SpeechPriority.SAFETY,
    ]
