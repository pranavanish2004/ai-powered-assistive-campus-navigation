"""
Campus graph domain model.

Design rationale
-----------------
This is pure domain logic: no database, no FastAPI, no I/O beyond the
explicit `to_dict`/`from_dict` serialization boundary. That's deliberate —
routing algorithms (Dijkstra/A*) should be testable with a graph built by
hand in a unit test in microseconds, with zero database or network
dependency.

The graph is treated as *authored, versioned data* (like a config file), not
user-mutable application state — it is loaded once via a repository (see
`interfaces/graph_interface.py`) and used read-only by the routing and
navigation layers for the lifetime of the process. Per-user navigation
progress lives in the database (see `infrastructure/database/models.py`),
never in this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from campus_nav.core.exceptions import EdgeNotFoundError, GraphValidationError, NodeNotFoundError


class NodeType(str, Enum):
    """Semantic category of a graph node. Affects how the instruction
    engine phrases guidance (e.g. "You have arrived at the entrance" vs
    "Continue to the next junction")."""

    BUILDING_ENTRANCE = "building_entrance"
    INTERSECTION = "intersection"
    LANDMARK = "landmark"
    ROOM = "room"
    WAYPOINT = "waypoint"
    AMENITY = "amenity"  # washroom, water fountain, etc.


class PathType(str, Enum):
    """Physical character of an edge. Drives both accessibility filtering
    and instruction phrasing ("ramp ahead" vs "corridor ahead")."""

    OUTDOOR_PATH = "outdoor_path"
    INDOOR_CORRIDOR = "indoor_corridor"
    ROAD_CROSSING = "road_crossing"
    STAIRS = "stairs"
    RAMP = "ramp"


@dataclass(frozen=True, slots=True)
class Node:
    """A single location in the campus graph."""

    id: str
    name: str
    latitude: float
    longitude: float
    node_type: NodeType = NodeType.WAYPOINT
    floor: int = 0
    is_accessible: bool = True
    landmark_description: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "node_type": self.node_type.value,
            "floor": self.floor,
            "is_accessible": self.is_accessible,
            "landmark_description": self.landmark_description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        return cls(
            id=data["id"],
            name=data["name"],
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            node_type=NodeType(data.get("node_type", NodeType.WAYPOINT.value)),
            floor=int(data.get("floor", 0)),
            is_accessible=bool(data.get("is_accessible", True)),
            landmark_description=data.get("landmark_description"),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """A connection between two nodes. Edges are bidirectional by default
    (`bidirectional=True`); a one-way edge (e.g. a one-way stairwell) sets
    it to False and is only traversable from_node -> to_node."""

    id: str
    from_node_id: str
    to_node_id: str
    distance_meters: float
    path_type: PathType = PathType.OUTDOOR_PATH
    is_accessible: bool = True
    bidirectional: bool = True
    landmark_description: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "distance_meters": self.distance_meters,
            "path_type": self.path_type.value,
            "is_accessible": self.is_accessible,
            "bidirectional": self.bidirectional,
            "landmark_description": self.landmark_description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        return cls(
            id=data["id"],
            from_node_id=data["from_node_id"],
            to_node_id=data["to_node_id"],
            distance_meters=float(data["distance_meters"]),
            path_type=PathType(data.get("path_type", PathType.OUTDOOR_PATH.value)),
            is_accessible=bool(data.get("is_accessible", True)),
            bidirectional=bool(data.get("bidirectional", True)),
            landmark_description=data.get("landmark_description"),
        )

    def other_end(self, node_id: str) -> str:
        """Given one endpoint, return the other. Raises ValueError if
        `node_id` is not an endpoint of this edge."""
        if node_id == self.from_node_id:
            return self.to_node_id
        if node_id == self.to_node_id:
            return self.from_node_id
        raise ValueError(f"Node {node_id!r} is not an endpoint of edge {self.id!r}.")


@dataclass
class CampusGraph:
    """An in-memory campus graph: nodes + edges + adjacency index.

    Construct via `add_node`/`add_edge`, or via `CampusGraph.from_dict` /
    `CampusGraph.load_json` for pre-authored data. Call `validate()` after
    building from untrusted input (e.g. a JSON file) — routing over an
    unvalidated graph could silently produce a nonsensical or unsafe route.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    _adjacency: dict[str, list[str]] = field(default_factory=dict, repr=False)

    # --- Construction ---
    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise GraphValidationError(f"Duplicate node id: {node.id!r}")
        self.nodes[node.id] = node
        self._adjacency.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        if edge.id in self.edges:
            raise GraphValidationError(f"Duplicate edge id: {edge.id!r}")
        if edge.from_node_id not in self.nodes:
            raise GraphValidationError(
                f"Edge {edge.id!r} references unknown from_node_id {edge.from_node_id!r}"
            )
        if edge.to_node_id not in self.nodes:
            raise GraphValidationError(
                f"Edge {edge.id!r} references unknown to_node_id {edge.to_node_id!r}"
            )
        if edge.distance_meters <= 0:
            raise GraphValidationError(f"Edge {edge.id!r} has non-positive distance.")

        self.edges[edge.id] = edge
        self._adjacency.setdefault(edge.from_node_id, []).append(edge.id)
        if edge.bidirectional:
            self._adjacency.setdefault(edge.to_node_id, []).append(edge.id)

    # --- Queries ---
    def get_node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise NodeNotFoundError(f"No such node: {node_id!r}") from None

    def get_edge(self, edge_id: str) -> Edge:
        try:
            return self.edges[edge_id]
        except KeyError:
            raise EdgeNotFoundError(f"No such edge: {edge_id!r}") from None

    def neighbors(self, node_id: str) -> list[tuple[Edge, Node]]:
        """Return (edge, neighboring_node) pairs reachable directly from
        `node_id`, respecting edge directionality."""
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"No such node: {node_id!r}")

        result: list[tuple[Edge, Node]] = []
        for edge_id in self._adjacency.get(node_id, []):
            edge = self.edges[edge_id]
            if edge.from_node_id == node_id:
                result.append((edge, self.nodes[edge.to_node_id]))
            elif edge.bidirectional and edge.to_node_id == node_id:
                result.append((edge, self.nodes[edge.from_node_id]))
        return result

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    # --- Validation ---
    def validate(self) -> list[str]:
        """Validate structural integrity. Raises GraphValidationError on any
        critical issue (dangling edge reference, duplicate id — though these
        are also caught eagerly in add_node/add_edge). Returns a list of
        non-fatal warnings (e.g. isolated nodes) for the caller to log."""
        warnings: list[str] = []

        for edge in self.edges.values():
            if edge.from_node_id not in self.nodes:
                raise GraphValidationError(
                    f"Edge {edge.id!r} references unknown from_node_id {edge.from_node_id!r}"
                )
            if edge.to_node_id not in self.nodes:
                raise GraphValidationError(
                    f"Edge {edge.id!r} references unknown to_node_id {edge.to_node_id!r}"
                )

        for node_id in self.nodes:
            if not self._adjacency.get(node_id):
                warnings.append(f"Node {node_id!r} has no edges (isolated).")

        return warnings

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CampusGraph:
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(Node.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(Edge.from_dict(edge_data))
        return graph
