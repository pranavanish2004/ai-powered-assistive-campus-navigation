"""
Mock GPS provider.

Real GPS hardware integration is deferred (and, per the architecture review,
GPS alone is insufficient indoors anyway — see Phase 1.2's localization
module for the snapping/indoor-fallback logic). This mock lets every layer
above it be built and tested today against a stable, controllable location
source rather than blocking on hardware. Swapping in a real NMEA/serial GPS
adapter later is a one-class change (see GPSInterface).
"""

from __future__ import annotations

from datetime import UTC, datetime

from campus_nav.interfaces.gps_interface import GPSCoordinate, GPSInterface

# Approximate coordinates of the IIT Hyderabad campus, used as a reasonable
# default for local development.
_DEFAULT_LATITUDE = 17.5937
_DEFAULT_LONGITUDE = 78.1245


class MockGPSProvider(GPSInterface):
    """Returns a fixed or manually-set coordinate. Useful for local
    development (no GPS hardware) and deterministic unit tests."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self._current = GPSCoordinate(
            latitude=_DEFAULT_LATITUDE,
            longitude=_DEFAULT_LONGITUDE,
            accuracy_meters=5.0,
            timestamp=datetime.now(UTC),
        )

    def set_location(self, latitude: float, longitude: float, accuracy_meters: float = 5.0) -> None:
        """Test/dev helper to simulate the user moving to a new position."""
        self._current = GPSCoordinate(
            latitude=latitude,
            longitude=longitude,
            accuracy_meters=accuracy_meters,
            timestamp=datetime.now(UTC),
        )

    def set_available(self, available: bool) -> None:
        """Test/dev helper to simulate loss of GPS fix (e.g. going indoors)."""
        self._available = available

    def get_current_location(self) -> GPSCoordinate | None:
        return self._current if self._available else None

    def is_available(self) -> bool:
        return self._available
