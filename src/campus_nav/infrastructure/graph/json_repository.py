"""JSON-file-backed campus graph repository."""

from __future__ import annotations

from pathlib import Path

from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.graph_interface import GraphRepositoryInterface
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.graph.serialization import load_graph_json, save_graph_json

logger = get_logger(__name__)


class JSONGraphRepository(GraphRepositoryInterface):
    """Loads/saves the campus graph from/to a JSON file on disk. This is the
    Phase 1.2 implementation; a database-backed or remote-config-backed
    repository could implement the same GraphRepositoryInterface later
    without any change to routing/navigation code."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> CampusGraph:
        graph = load_graph_json(self._path)
        warnings = graph.validate()
        for warning in warnings:
            logger.warning("Graph validation warning", extra={"warning": warning})
        return graph

    def save(self, graph: CampusGraph) -> None:
        graph.validate()
        save_graph_json(graph, self._path)
