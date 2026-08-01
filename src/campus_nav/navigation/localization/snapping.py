"""
GPS-to-graph snapping.

A raw (lat, lon) fix rarely lands exactly on a graph node. Snapping finds
the most plausible place on the campus graph the user is actually standing
— either a specific node, or a point along an edge between two nodes — so
the Navigation Engine can reason in graph terms ("you are between node A and
node B") rather than raw coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

from campus_nav.core.exceptions import SnappingFailedError
from campus_nav.navigation.graph.geometry import (
    haversine_distance_meters,
    project_point_onto_segment,
)
from campus_nav.navigation.graph.models import CampusGraph, Edge, Node


@dataclass(frozen=True, slots=True)
class SnapResult:
    """Where on the campus graph a raw GPS fix most plausibly corresponds
    to. Exactly one of `nearest_node`/`nearest_edge` is the "primary" match
    (whichever had the smaller snapping distance); both distances are
    reported so callers can make their own threshold decisions if needed."""

    snapped_latitude: float
    snapped_longitude: float
    distance_meters: float
    nearest_node: Node
    nearest_edge: Edge | None
    """None only when the graph has zero edges (a degenerate/empty graph)."""


def find_nearest_node(graph: CampusGraph, latitude: float, longitude: float) -> tuple[Node, float]:
    """Return the (node, distance_meters) pair for the graph node closest
    to the given coordinate. Raises SnappingFailedError if the graph has no
    nodes at all."""
    if not graph.nodes:
        raise SnappingFailedError("Cannot snap: campus graph has no nodes.")

    best_node: Node | None = None
    best_distance = float("inf")
    for node in graph.nodes.values():
        distance = haversine_distance_meters(latitude, longitude, node.latitude, node.longitude)
        if distance < best_distance:
            best_distance = distance
            best_node = node

    assert best_node is not None  # graph.nodes is non-empty, guaranteed above
    return best_node, best_distance


def find_nearest_edge(
    graph: CampusGraph, latitude: float, longitude: float
) -> tuple[Edge, float, float, float] | None:
    """Return (edge, projected_lat, projected_lon, distance_meters) for the
    graph edge whose segment is closest to the given coordinate, or None if
    the graph has no edges."""
    if not graph.edges:
        return None

    best: tuple[Edge, float, float, float] | None = None
    best_distance = float("inf")

    for edge in graph.edges.values():
        node_a = graph.get_node(edge.from_node_id)
        node_b = graph.get_node(edge.to_node_id)
        proj_lat, proj_lon, distance = project_point_onto_segment(
            latitude,
            longitude,
            node_a.latitude,
            node_a.longitude,
            node_b.latitude,
            node_b.longitude,
        )
        if distance < best_distance:
            best_distance = distance
            best = (edge, proj_lat, proj_lon, distance)

    return best


def snap_to_graph(
    graph: CampusGraph,
    latitude: float,
    longitude: float,
    max_distance_meters: float,
) -> SnapResult:
    """Snap a raw GPS fix onto the campus graph.

    Considers both the nearest node and the nearest point along any edge,
    and returns whichever is closer. Raises SnappingFailedError if even the
    best match exceeds `max_distance_meters` — beyond that distance, trusting
    the snap would be worse than admitting the position is unknown (see
    Settings.gps_snap_max_distance_meters and the Localizer's handling of
    this exception as a trigger for the NavigationEngine's UNCERTAIN state).
    """
    nearest_node, node_distance = find_nearest_node(graph, latitude, longitude)
    edge_match = find_nearest_edge(graph, latitude, longitude)

    if edge_match is not None and edge_match[3] < node_distance:
        edge, proj_lat, proj_lon, edge_distance = edge_match
        if edge_distance > max_distance_meters:
            raise SnappingFailedError(
                f"Nearest graph element is {edge_distance:.1f}m away, "
                f"exceeding max snap distance of {max_distance_meters:.1f}m."
            )
        return SnapResult(
            snapped_latitude=proj_lat,
            snapped_longitude=proj_lon,
            distance_meters=edge_distance,
            nearest_node=nearest_node,
            nearest_edge=edge,
        )

    if node_distance > max_distance_meters:
        raise SnappingFailedError(
            f"Nearest graph element is {node_distance:.1f}m away, "
            f"exceeding max snap distance of {max_distance_meters:.1f}m."
        )

    return SnapResult(
        snapped_latitude=nearest_node.latitude,
        snapped_longitude=nearest_node.longitude,
        distance_meters=node_distance,
        nearest_node=nearest_node,
        nearest_edge=edge_match[0] if edge_match else None,
    )
