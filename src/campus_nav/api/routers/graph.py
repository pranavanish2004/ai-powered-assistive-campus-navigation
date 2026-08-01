"""Graph inspection endpoints -- primarily for debugging, dashboards, and
integration tests. The campus graph is read-only from the API's
perspective (see api/dependencies.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from campus_nav.api.dependencies import get_graph
from campus_nav.api.schemas.graph import EdgeResponse, GraphResponse, NodeResponse
from campus_nav.core.exceptions import NodeNotFoundError
from campus_nav.navigation.graph.models import CampusGraph

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
def get_full_graph(graph: CampusGraph = Depends(get_graph)) -> GraphResponse:
    return GraphResponse.from_domain(graph)


@router.get("/nodes/{node_id}", response_model=NodeResponse)
def get_node(node_id: str, graph: CampusGraph = Depends(get_graph)) -> NodeResponse:
    try:
        node = graph.get_node(node_id)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NodeResponse.from_domain(node)


@router.get("/nodes/{node_id}/neighbors", response_model=list[EdgeResponse])
def get_node_neighbors(node_id: str, graph: CampusGraph = Depends(get_graph)) -> list[EdgeResponse]:
    try:
        neighbors = graph.neighbors(node_id)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [EdgeResponse.from_domain(edge) for edge, _ in neighbors]
