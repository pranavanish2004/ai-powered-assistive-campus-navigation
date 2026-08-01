"""Pydantic request/response schemas for the graph endpoints.

These are deliberately separate from the domain `Node`/`Edge` dataclasses
(navigation/graph/models.py) -- the API's wire format and the domain
model are allowed to evolve independently (e.g. hiding an internal-only
field from API consumers) without one constraining the other.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from campus_nav.navigation.graph.models import CampusGraph, Edge, NodeType, PathType
from campus_nav.navigation.graph.models import Node as DomainNode


class NodeResponse(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    node_type: NodeType
    floor: int
    is_accessible: bool
    landmark_description: str | None = None

    @classmethod
    def from_domain(cls, node: DomainNode) -> NodeResponse:
        return cls(
            id=node.id,
            name=node.name,
            latitude=node.latitude,
            longitude=node.longitude,
            node_type=node.node_type,
            floor=node.floor,
            is_accessible=node.is_accessible,
            landmark_description=node.landmark_description,
        )


class EdgeResponse(BaseModel):
    id: str
    from_node_id: str
    to_node_id: str
    distance_meters: float
    path_type: PathType
    is_accessible: bool
    bidirectional: bool
    landmark_description: str | None = None

    @classmethod
    def from_domain(cls, edge: Edge) -> EdgeResponse:
        return cls(
            id=edge.id,
            from_node_id=edge.from_node_id,
            to_node_id=edge.to_node_id,
            distance_meters=edge.distance_meters,
            path_type=edge.path_type,
            is_accessible=edge.is_accessible,
            bidirectional=edge.bidirectional,
            landmark_description=edge.landmark_description,
        )


class GraphResponse(BaseModel):
    nodes: list[NodeResponse] = Field(default_factory=list)
    edges: list[EdgeResponse] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, graph: CampusGraph) -> GraphResponse:
        return cls(
            nodes=[NodeResponse.from_domain(n) for n in graph.nodes.values()],
            edges=[EdgeResponse.from_domain(e) for e in graph.edges.values()],
        )
