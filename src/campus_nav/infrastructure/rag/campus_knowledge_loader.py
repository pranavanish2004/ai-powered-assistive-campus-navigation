"""
Campus knowledge loader.

Combines two sources into one knowledge base, per the Phase 1.4a design:

1. Every Campus Graph node's `landmark_description` (Phase 1.2 data,
   already authored for exactly this kind of narrative use) becomes a
   `KnowledgeDocument` automatically -- no duplicate authoring effort.
2. A static JSON file (`data/campus_knowledge.json`) supplies facts the
   graph has no field for: accessibility overviews, emergency contacts,
   campus rules -- exactly the categories the original specification
   named that aren't naturally "a location on a graph."

This keeps the campus graph the single source of truth for *where* things
are, and the static file the source of truth for facts that aren't a
place at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from campus_nav.core.exceptions import RAGError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.rag_interface import (
    KnowledgeCategory,
    KnowledgeDocument,
    RAGRepositoryInterface,
)
from campus_nav.navigation.graph.models import CampusGraph, NodeType

logger = get_logger(__name__)

_NODE_TYPE_TO_CATEGORY: dict[NodeType, KnowledgeCategory] = {
    NodeType.BUILDING_ENTRANCE: KnowledgeCategory.BUILDING,
    NodeType.LANDMARK: KnowledgeCategory.BUILDING,
    NodeType.AMENITY: KnowledgeCategory.GENERAL,
}


def documents_from_graph(graph: CampusGraph) -> list[KnowledgeDocument]:
    """Build one KnowledgeDocument per graph node that has a
    `landmark_description`. Nodes without one (e.g. bare waypoints/
    intersections with no narrative value) are skipped -- they exist for
    routing, not for conversation."""
    documents: list[KnowledgeDocument] = []
    for node in graph.nodes.values():
        if not node.landmark_description:
            continue
        category = _NODE_TYPE_TO_CATEGORY.get(node.node_type, KnowledgeCategory.GENERAL)
        documents.append(
            KnowledgeDocument(
                id=f"graph_node_{node.id}",
                title=node.name,
                content=node.landmark_description,
                category=category,
                metadata={"node_id": node.id, "is_accessible": node.is_accessible},
                source="campus_graph",
            )
        )
    return documents


def documents_from_static_file(path: str | Path) -> list[KnowledgeDocument]:
    """Load supplementary knowledge documents from a JSON file. Returns an
    empty list (with a logged warning) if the file is missing or
    malformed -- the knowledge base should degrade gracefully, not prevent
    application startup, if this optional supplementary file is absent."""
    path = Path(path)
    if not path.exists():
        logger.warning("Static knowledge file not found; skipping", extra={"path": str(path)})
        return []

    try:
        raw_entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to load static knowledge file; skipping",
            extra={"path": str(path), "error": str(exc)},
        )
        return []

    documents: list[KnowledgeDocument] = []
    default_source = f"static_knowledge:{path.name}"
    for entry in raw_entries:
        try:
            documents.append(
                KnowledgeDocument(
                    id=entry["id"],
                    title=entry["title"],
                    content=entry["content"],
                    category=KnowledgeCategory(entry.get("category", "general")),
                    metadata=entry.get("metadata", {}),
                    source=entry.get("source", default_source),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Skipping malformed knowledge entry",
                extra={"entry": str(entry)[:200], "error": str(exc)},
            )
    return documents


def build_campus_knowledge_base(
    repository: RAGRepositoryInterface,
    graph: CampusGraph,
    static_knowledge_path: str | Path,
) -> RAGRepositoryInterface:
    """Populate `repository` with documents derived from `graph` and the
    static knowledge file, and return it (for convenient chaining)."""
    try:
        for document in documents_from_graph(graph):
            repository.add_document(document)
        for document in documents_from_static_file(static_knowledge_path):
            repository.add_document(document)
    except Exception as exc:
        raise RAGError(f"Failed to build campus knowledge base: {exc}") from exc

    logger.info("Campus knowledge base built", extra={"documents": repository.document_count()})
    return repository
