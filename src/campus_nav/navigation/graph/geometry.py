"""
Geographic math utilities shared across routing, localization, and
instruction generation.

Kept in one place so the whole system uses a single, consistent notion of
"distance between two coordinates" — a hand-rolled haversine formula copied
into three different modules is exactly the kind of duplication that leads
to subtle A*-heuristic-is-not-actually-admissible bugs later.
"""

from __future__ import annotations

import math

_EARTH_RADIUS_METERS = 6_371_000.0


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in meters.

    Used as the A* heuristic (straight-line distance is always <= any real
    path distance across campus, so the heuristic is admissible and A*
    remains optimal) and as the basis for GPS-to-graph snapping distance.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_METERS * c


def initial_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (0-360, 0=North) from point 1 to point 2. Used by the
    turn-by-turn instruction engine to classify turns (left/right/straight)
    from the change in bearing between consecutive edges."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)

    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def bearing_difference_degrees(bearing_from: float, bearing_to: float) -> float:
    """Signed difference between two bearings, normalized to (-180, 180].
    Positive = turn right/clockwise, negative = turn left/counter-clockwise."""
    diff = (bearing_to - bearing_from + 180) % 360 - 180
    return diff


def project_point_onto_segment(
    point_lat: float,
    point_lon: float,
    seg_a_lat: float,
    seg_a_lon: float,
    seg_b_lat: float,
    seg_b_lon: float,
) -> tuple[float, float, float]:
    """Project a point onto the line segment A-B using a local equirectangular
    approximation (accurate for the sub-kilometer distances relevant to a
    single campus). Returns (projected_lat, projected_lon, distance_meters
    from the original point to the projection).

    This is deliberately an approximation, not a full geodesic projection —
    campus-scale distances (tens to low-hundreds of meters) make the flat-
    earth approximation error negligible (sub-centimeter), while keeping the
    math simple enough to unit-test exactly.
    """
    # Convert to a local planar (x=east, y=north) approximation in meters,
    # anchored at segment point A.
    lat_to_m = 111_320.0  # meters per degree latitude, ~constant
    lon_to_m = 111_320.0 * math.cos(math.radians(seg_a_lat))

    ax, ay = 0.0, 0.0
    bx = (seg_b_lon - seg_a_lon) * lon_to_m
    by = (seg_b_lat - seg_a_lat) * lat_to_m
    px = (point_lon - seg_a_lon) * lon_to_m
    py = (point_lat - seg_a_lat) * lat_to_m

    abx, aby = bx - ax, by - ay
    seg_len_sq = abx**2 + aby**2

    if seg_len_sq == 0:
        # Degenerate segment (A == B); projection is just point A.
        t = 0.0
    else:
        t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
        t = max(0.0, min(1.0, t))  # clamp to the segment, not the infinite line

    proj_x = ax + t * abx
    proj_y = ay + t * aby

    proj_lon = seg_a_lon + proj_x / lon_to_m
    proj_lat = seg_a_lat + proj_y / lat_to_m

    distance = math.hypot(px - proj_x, py - proj_y)
    return proj_lat, proj_lon, distance
