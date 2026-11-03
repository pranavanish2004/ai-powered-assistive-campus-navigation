"""
Campus data ingestion abstraction.

Design rationale
-----------------
The Navigation Graph (`navigation.graph.models.CampusGraph`) and the RAG
knowledge base (`interfaces.rag_interface.RAGRepositoryInterface`) are
intentionally **not modified** by this feature -- per the explicit
requirement to keep that architecture unchanged. Instead, importers here
produce plain, format-agnostic value objects (`ImportedNode`,
`ImportedEdge`, `ImportedKnowledgeEntry`) that `ingestion_pipeline.py`
converts into ordinary `Node`/`Edge`/`KnowledgeDocument` calls against the
existing, unmodified `CampusGraph.add_node/add_edge` and
`RAGRepositoryInterface.add_document` APIs. Adding a new source format
means adding a new `CampusDataImporterInterface` implementation -- it
never means touching the graph or knowledge-base classes themselves.

This is what "no code changes required when new campus data is added"
concretely means: dropping a new `.geojson`/`.csv`/`.yaml`/`.md` file into
the configured data directory is a data change, not a code change, because
every format already has a registered importer before that file ever
arrives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImportedNode:
    """Format-agnostic representation of one location, before being
    converted into a `navigation.graph.models.Node`."""

    id: str
    name: str
    latitude: float
    longitude: float
    node_type: str = "waypoint"
    floor: int = 0
    is_accessible: bool = True
    landmark_description: str | None = None
    aliases: list[str] = field(default_factory=list)
    """Alternate names a user might say, e.g. ["Bio Building", "BME"] for
    "Biomedical Engineering Building" -- not used by the Campus Graph
    (which has no alias field), but forwarded into the knowledge base's
    document content so a query using an alias can still retrieve the
    right fact. See ingestion_pipeline.py for exactly how."""


@dataclass(frozen=True, slots=True)
class ImportedEdge:
    """Format-agnostic representation of one path segment, before being
    converted into a `navigation.graph.models.Edge`."""

    id: str
    from_node_id: str
    to_node_id: str
    distance_meters: float | None = None
    """None means "compute from the two endpoints' coordinates" -- see
    ingestion_pipeline.py. Explicit values are honored as-is, since a real
    path is often longer than the straight-line distance between its
    endpoints."""
    path_type: str = "outdoor_path"
    is_accessible: bool = True
    bidirectional: bool = True
    landmark_description: str | None = None


@dataclass(frozen=True, slots=True)
class ImportedKnowledgeEntry:
    """Format-agnostic representation of one RAG fact, before being
    converted into an `interfaces.rag_interface.KnowledgeDocument`."""

    id: str
    title: str
    content: str
    category: str = "general"
    metadata: dict = field(default_factory=dict)


@dataclass
class ImportedCampusData:
    """The combined output of one or more importers, ready to be merged
    into an existing CampusGraph and knowledge base."""

    nodes: list[ImportedNode] = field(default_factory=list)
    edges: list[ImportedEdge] = field(default_factory=list)
    knowledge_entries: list[ImportedKnowledgeEntry] = field(default_factory=list)

    def merge(self, other: ImportedCampusData) -> None:
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
        self.knowledge_entries.extend(other.knowledge_entries)


class CampusDataImporterInterface(ABC):
    """Abstract base class for one source-format importer."""

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this importer knows how to read `path` (usually
        a file-extension check). Lets the ingestion pipeline route each
        file in a data directory to the correct importer without a
        hardcoded if/elif chain that would need editing for new formats."""

    @abstractmethod
    def import_data(self, path: Path) -> ImportedCampusData:
        """Parse `path` and return the campus data it contains.
        Implementations should raise IngestionError (core.exceptions) on
        malformed input rather than silently returning partial/incorrect
        data, since incorrect coordinates in a navigation-for-the-blind
        system are a safety concern, not just a data-quality one."""
