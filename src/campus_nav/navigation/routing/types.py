"""Shared types used by every routing algorithm (Dijkstra, A*, multi-route)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from campus_nav.navigation.graph.models import CampusGraph, Edge


class AccessibilityMode(str, Enum):
    """How strictly a route must avoid non-accessible edges (e.g. stairs).

    - NONE: accessibility is ignored; shortest path only.
    - PREFERRED: accessible edges are strongly favored via a weight penalty
      on inaccessible edges, but an inaccessible edge may still be used if
      it is the only way to reach the destination (better to route the
      user with a warning than report "no route found" when a route
      technically exists).
    - STRICT: inaccessible edges are excluded entirely. Raises
      NoRouteFoundError if no all-accessible path exists.
    """

    NONE = "none"
    PREFERRED = "preferred"
    STRICT = "strict"


# Multiplier applied to an inaccessible edge's distance under PREFERRED mode.
# Large enough that any accessible alternative is chosen over it, but finite
# so an inaccessible edge remains usable as a last resort.
_PREFERRED_MODE_PENALTY = 20.0


def edge_traversal_weight(edge: Edge, mode: AccessibilityMode) -> float:
    """Return the routing weight for `edge` under the given accessibility
    mode, or math.inf if the edge must not be used at all (STRICT mode,
    inaccessible edge)."""
    if edge.is_accessible or mode is AccessibilityMode.NONE:
        return edge.distance_meters
    if mode is AccessibilityMode.STRICT:
        return math.inf
    return edge.distance_meters * _PREFERRED_MODE_PENALTY  # PREFERRED


@dataclass(frozen=True, slots=True)
class Route:
    """The result of a successful routing computation."""

    node_ids: list[str]
    edge_ids: list[str]
    total_distance_meters: float
    algorithm: str
    accessibility_mode: AccessibilityMode
    uses_inaccessible_edge: bool

    def is_empty(self) -> bool:
        return len(self.node_ids) < 2

    @staticmethod
    def from_path(
        graph: CampusGraph,
        node_ids: list[str],
        edge_ids: list[str],
        algorithm: str,
        accessibility_mode: AccessibilityMode,
    ) -> Route:
        total_distance = sum(graph.get_edge(eid).distance_meters for eid in edge_ids)
        uses_inaccessible = any(not graph.get_edge(eid).is_accessible for eid in edge_ids)
        return Route(
            node_ids=node_ids,
            edge_ids=edge_ids,
            total_distance_meters=total_distance,
            algorithm=algorithm,
            accessibility_mode=accessibility_mode,
            uses_inaccessible_edge=uses_inaccessible,
        )
