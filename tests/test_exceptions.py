"""Tests asserting every module-specific exception is correctly classified
as RecoverableError or FatalError. This classification is load-bearing:
calling code in the Decision Engine branches on it (see exceptions.py
docstring), so a misclassified exception is a real safety bug, not just a
style issue."""

import pytest

from campus_nav.core.exceptions import (
    CameraError,
    CameraUnavailableError,
    ConfigurationError,
    DatabaseError,
    FatalError,
    GPSError,
    RecoverableError,
    SpeechRecognitionError,
    TextToSpeechError,
)


@pytest.mark.parametrize(
    "exc_class",
    [CameraError, GPSError, SpeechRecognitionError],
)
def test_recoverable_exceptions_are_classified_correctly(exc_class):
    assert issubclass(exc_class, RecoverableError)
    assert not issubclass(exc_class, FatalError)


@pytest.mark.parametrize(
    "exc_class",
    [ConfigurationError, CameraUnavailableError, TextToSpeechError, DatabaseError],
)
def test_fatal_exceptions_are_classified_correctly(exc_class):
    assert issubclass(exc_class, FatalError)


def test_exceptions_carry_their_message():
    exc = CameraError("device busy")
    assert str(exc) == "device busy"
