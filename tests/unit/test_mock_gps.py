"""Tests for MockGPSProvider."""

import pytest

from campus_nav.infrastructure.gps.mock_gps import MockGPSProvider

pytestmark = pytest.mark.unit


def test_default_location_is_iith_campus():
    gps = MockGPSProvider()
    location = gps.get_current_location()
    assert location is not None
    assert 17.0 < location.latitude < 18.0
    assert 78.0 < location.longitude < 79.0


def test_set_location_updates_fix():
    gps = MockGPSProvider()
    gps.set_location(1.0, 2.0, accuracy_meters=3.0)
    location = gps.get_current_location()
    assert location.latitude == 1.0
    assert location.longitude == 2.0
    assert location.accuracy_meters == 3.0


def test_set_available_false_simulates_indoor_loss_of_fix():
    gps = MockGPSProvider()
    gps.set_available(False)
    assert gps.get_current_location() is None
    assert gps.is_available() is False


def test_starts_available_by_default():
    gps = MockGPSProvider()
    assert gps.is_available() is True


def test_can_start_unavailable():
    gps = MockGPSProvider(available=False)
    assert gps.get_current_location() is None
