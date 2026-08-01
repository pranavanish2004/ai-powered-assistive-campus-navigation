"""
Graph JSON serialization.

Kept separate from `models.py`'s in-memory `to_dict`/`from_dict` so the
domain model has zero file-I/O dependency — this module is the only place
that touches a filesystem path or raw JSON text, per Clean Architecture's
separation of domain logic from I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

from campus_nav.core.exceptions import GraphSerializationError
from campus_nav.core.logging_config import get_logger
from campus_nav.navigation.graph.models import CampusGraph

logger = get_logger(__name__)


def save_graph_json(graph: CampusGraph, path: str | Path) -> None:
    """Serialize `graph` to a JSON file at `path`."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "Graph saved",
            extra={"path": str(path), "nodes": graph.node_count(), "edges": graph.edge_count()},
        )
    except OSError as exc:
        raise GraphSerializationError(f"Failed to write graph to {path}: {exc}") from exc


def load_graph_json(path: str | Path) -> CampusGraph:
    """Deserialize a CampusGraph from a JSON file at `path`.

    Note: this only performs structural loading (delegated to
    `CampusGraph.from_dict`, which already validates node/edge references
    eagerly via add_node/add_edge). Callers should still call
    `graph.validate()` afterward to surface non-fatal warnings such as
    isolated nodes.
    """
    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphSerializationError(f"Failed to read graph file {path}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise GraphSerializationError(f"Malformed graph JSON in {path}: {exc}") from exc

    graph = CampusGraph.from_dict(data)
    logger.info(
        "Graph loaded",
        extra={"path": str(path), "nodes": graph.node_count(), "edges": graph.edge_count()},
    )
    return graph


def loads_graph_json(raw_text: str) -> CampusGraph:
    """Deserialize a CampusGraph from a JSON string (no filesystem access).
    Useful for tests and for graphs embedded in API request bodies."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise GraphSerializationError(f"Malformed graph JSON: {exc}") from exc
    return CampusGraph.from_dict(data)
