"""
Localization confidence estimation.

A single "confidence" score in [0.0, 1.0] lets the Safety Engine and
Navigation Engine make decisions (trust this position? warn the user?
declare UNCERTAIN state?) without every caller re-deriving its own
threshold logic from raw distances. Confidence degrades with both:

1. Raw GPS receiver accuracy (a wide accuracy_meters radius means the
   receiver itself is unsure).
2. Snapping distance (how far the raw fix was from any known graph
   element — a large snap distance means the graph match itself is
   questionable, e.g. the user could be in an unmapped area).
"""

from __future__ import annotations


def confidence_from_snap_distance(distance_meters: float, max_distance_meters: float) -> float:
    """Linear falloff: confidence is 1.0 at zero distance, 0.0 at
    `max_distance_meters`, and clamped to [0, 1] beyond that range."""
    if max_distance_meters <= 0:
        return 0.0
    confidence = 1.0 - (distance_meters / max_distance_meters)
    return max(0.0, min(1.0, confidence))


def confidence_from_gps_accuracy(
    accuracy_meters: float, reference_accuracy_meters: float = 10.0
) -> float:
    """Linear falloff based on the GPS receiver's own reported accuracy
    radius. `reference_accuracy_meters` is the accuracy at which confidence
    reaches zero (typical consumer GPS accuracy indoors/urban-canyon can
    easily exceed 10m, which is why this is a fairly tight reference)."""
    if reference_accuracy_meters <= 0:
        return 0.0
    confidence = 1.0 - (accuracy_meters / reference_accuracy_meters)
    return max(0.0, min(1.0, confidence))


def combined_localization_confidence(
    snap_distance_meters: float,
    max_snap_distance_meters: float,
    gps_accuracy_meters: float,
    reference_gps_accuracy_meters: float = 10.0,
) -> float:
    """Combine both factors conservatively (take the minimum, not the
    average) — localization is only as trustworthy as its weakest input.
    A precise GPS fix that snaps far from any known path should still be
    treated with suspicion, and vice versa."""
    snap_confidence = confidence_from_snap_distance(snap_distance_meters, max_snap_distance_meters)
    gps_confidence = confidence_from_gps_accuracy(
        gps_accuracy_meters, reference_gps_accuracy_meters
    )
    return min(snap_confidence, gps_confidence)
