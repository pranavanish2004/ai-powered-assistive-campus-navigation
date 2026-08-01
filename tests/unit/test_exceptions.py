"""Tests asserting every exception is classified as EXACTLY one of
RecoverableError/FatalError -- see core/exceptions.py module docstring on
why "exactly one branch" is a hard constraint, not a style preference."""

import inspect

import pytest

from campus_nav.core import exceptions as exc_module
from campus_nav.core.exceptions import CampusNavError, FatalError, RecoverableError

pytestmark = pytest.mark.unit


def _all_campus_nav_exceptions():
    for _, obj in inspect.getmembers(exc_module, inspect.isclass):
        if issubclass(obj, CampusNavError) and obj is not CampusNavError:
            yield obj


@pytest.mark.parametrize("exc_class", list(_all_campus_nav_exceptions()))
def test_every_exception_is_recoverable_xor_fatal(exc_class):
    is_recoverable = issubclass(exc_class, RecoverableError)
    is_fatal = issubclass(exc_class, FatalError)
    assert is_recoverable or is_fatal, f"{exc_class.__name__} is neither Recoverable nor Fatal"
    assert not (is_recoverable and is_fatal), (
        f"{exc_class.__name__} is BOTH Recoverable and Fatal -- this is exactly "
        "the MRO-conflict-prone pattern the hierarchy is designed to prevent."
    )


@pytest.mark.parametrize(
    "exc_class",
    [
        exc_module.CameraError,
        exc_module.GPSError,
        exc_module.SpeechRecognitionError,
        exc_module.GraphError,
        exc_module.NodeNotFoundError,
        exc_module.EdgeNotFoundError,
        exc_module.GraphSerializationError,
        exc_module.RoutingError,
        exc_module.NoRouteFoundError,
        exc_module.LocalizationError,
        exc_module.SnappingFailedError,
        exc_module.NavigationError,
    ],
)
def test_recoverable_classification(exc_class):
    assert issubclass(exc_class, RecoverableError)


@pytest.mark.parametrize(
    "exc_class",
    [
        exc_module.ConfigurationError,
        exc_module.CameraUnavailableError,
        exc_module.TextToSpeechError,
        exc_module.DatabaseError,
        exc_module.GraphValidationError,
        exc_module.InvalidStateTransitionError,
    ],
)
def test_fatal_classification(exc_class):
    assert issubclass(exc_class, FatalError)


def test_exceptions_carry_their_message():
    exc = exc_module.CameraError("device busy")
    assert str(exc) == "device busy"


def test_no_mro_conflicts_on_instantiation():
    """The bug this hierarchy was designed to prevent: instantiating every
    leaf exception must not raise TypeError from an unresolvable MRO."""
    for exc_class in _all_campus_nav_exceptions():
        instance = exc_class("test message")
        assert isinstance(instance, CampusNavError)
