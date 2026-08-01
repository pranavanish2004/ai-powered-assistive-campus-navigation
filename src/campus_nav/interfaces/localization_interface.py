"""
Localization abstraction.

Wraps "where is the user, in terms the campus graph understands" behind one
interface. Today's implementation combines GPSInterface + graph snapping;
an indoor-localization implementation (BLE beacons, WiFi RTT, or
landmark-inferred position from the Phase 1.3 vision pipeline) can be added
later as a second implementation, or composed with this one, without the
Navigation Engine changing at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from campus_nav.navigation.graph.models import CampusGraph, Edge, Node


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """The Navigation Engine's view of "where is the user right now,"
    expressed in graph terms rather than raw coordinates."""

    latitude: float
    longitude: float
    nearest_node: Node
    nearest_edge: Edge | None
    snap_distance_meters: float
    confidence: float
    """In [0.0, 1.0]. See navigation/localization/confidence.py."""


class LocalizationInterface(ABC):
    """Abstract base class for producing a LocalizationResult for the
    current moment, given a campus graph to snap against."""

    @abstractmethod
    def get_current_position(self, graph: CampusGraph) -> LocalizationResult | None:
        """Return the current LocalizationResult, or None if position
        cannot be determined at all right now (e.g. GPS has no fix and no
        other localization source is available). Returning None is a
        normal, expected outcome the caller must handle — it is not raised
        as an exception, because "unknown position" is routine when a user
        first steps indoors."""
