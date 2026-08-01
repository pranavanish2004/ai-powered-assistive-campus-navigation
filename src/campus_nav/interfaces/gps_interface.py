"""
GPS / outdoor localization abstraction.

GPS is only one localization source among several (see the Localization
interface / indoor fallback in Phase 1.2). Hiding it behind this interface
lets a future indoor provider be combined via a composite provider without
touching Navigation Engine code.

`accuracy_meters` is part of the value object deliberately — the caller must
be able to say "this fix is too imprecise to trust" rather than accepting a
noisy fix blindly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class GPSCoordinate:
    """Immutable value object representing a single position fix."""

    latitude: float
    longitude: float
    accuracy_meters: float
    timestamp: datetime


class GPSInterface(ABC):
    """Abstract base class for any outdoor positioning source."""

    @abstractmethod
    def get_current_location(self) -> GPSCoordinate | None:
        """Return the most recent position fix, or None if no fix is
        currently available (e.g. indoors, cold start, no satellite lock).
        "No fix" is expected, common (especially indoors) control flow —
        not an exceptional error path."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider currently has a usable fix."""
