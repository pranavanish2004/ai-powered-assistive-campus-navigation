"""
GraphSnappingLocalizer: the default LocalizationInterface implementation.

Combines a GPSInterface source with graph snapping (nearest node/edge) and
confidence scoring into a single LocalizationResult. This is where "GPS is
unreliable indoors" (flagged in the Phase 1.1 review) becomes concrete
handled behavior: when the GPS source reports no fix, or the fix cannot be
snapped within the configured maximum distance, this class returns None
rather than fabricating a guess — the Navigation Engine treats that as a
trigger to enter the UNCERTAIN state, not as a crash.
"""

from __future__ import annotations

from campus_nav.core.exceptions import SnappingFailedError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.gps_interface import GPSInterface
from campus_nav.interfaces.localization_interface import LocalizationInterface, LocalizationResult
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.localization.confidence import combined_localization_confidence
from campus_nav.navigation.localization.snapping import snap_to_graph

logger = get_logger(__name__)


def localize_fix(
    graph: CampusGraph,
    latitude: float,
    longitude: float,
    accuracy_meters: float,
    max_snap_distance_meters: float,
) -> LocalizationResult | None:
    """Snap a raw (latitude, longitude, accuracy) fix onto `graph` and
    compute its confidence. Returns None if the fix cannot be snapped
    within `max_snap_distance_meters` — this is the single shared
    implementation used both by GraphSnappingLocalizer (for callers that
    poll a GPSInterface themselves) and directly by NavigationEngine (for
    callers, like the API layer, that already have a raw fix in hand and
    should not need to go through a GPSInterface indirection to use it).
    """
    try:
        snap = snap_to_graph(graph, latitude, longitude, max_snap_distance_meters)
    except SnappingFailedError as exc:
        logger.info("Position fix could not be snapped to campus graph", extra={"reason": str(exc)})
        return None

    confidence = combined_localization_confidence(
        snap_distance_meters=snap.distance_meters,
        max_snap_distance_meters=max_snap_distance_meters,
        gps_accuracy_meters=accuracy_meters,
    )

    return LocalizationResult(
        latitude=snap.snapped_latitude,
        longitude=snap.snapped_longitude,
        nearest_node=snap.nearest_node,
        nearest_edge=snap.nearest_edge,
        snap_distance_meters=snap.distance_meters,
        confidence=confidence,
    )


class GraphSnappingLocalizer(LocalizationInterface):
    """LocalizationInterface implementation for callers that want the
    engine to pull position from a GPSInterface itself (e.g. a background
    daemon polling local GPS hardware), rather than supplying coordinates
    directly. NavigationEngine.update_position() does NOT use this class —
    it receives coordinates explicitly (e.g. from a mobile client via the
    API) and calls `localize_fix` directly. This class exists for the
    alternate embedded/hardware-polling usage pattern."""

    def __init__(self, gps: GPSInterface, max_snap_distance_meters: float = 30.0) -> None:
        self._gps = gps
        self._max_snap_distance_meters = max_snap_distance_meters

    def get_current_position(self, graph: CampusGraph) -> LocalizationResult | None:
        fix = self._gps.get_current_location()
        if fix is None:
            logger.debug("No GPS fix available")
            return None

        return localize_fix(
            graph, fix.latitude, fix.longitude, fix.accuracy_meters, self._max_snap_distance_meters
        )
