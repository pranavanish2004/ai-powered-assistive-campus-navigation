"""
Campus data ingestion pipeline.

This is the single place that (a) knows every registered importer, (b)
scans a directory and routes each file to the importer that claims it,
and (c) converts the resulting format-agnostic `ImportedCampusData` into
ordinary calls against the existing, unmodified `CampusGraph` and
`RAGRepositoryInterface` -- see `interfaces/ingestion_interface.py`'s
docstring for why that conversion boundary is exactly where it is.

Adding a new format
---------------------
Register a new `CampusDataImporterInterface` implementation in
`default_importers()` below. Nothing else in this file, the graph, or the
knowledge base needs to change -- this is what "no code changes required
when new campus data is added" means for *formats*; for *data*, it means
literally just adding a file to the configured directory.
"""

from __future__ import annotations

from pathlib import Path

from campus_nav.core.exceptions import IngestionError
from campus_nav.core.logging_config import get_logger
from campus_nav.infrastructure.ingestion.csv_importer import CSVImporter
from campus_nav.infrastructure.ingestion.geojson_importer import GeoJSONImporter
from campus_nav.infrastructure.ingestion.markdown_importer import MarkdownImporter
from campus_nav.infrastructure.ingestion.osm_importer import OSMPBFImporter, OSMXMLImporter
from campus_nav.infrastructure.ingestion.pdf_importer import PDFImporter
from campus_nav.infrastructure.ingestion.yaml_importer import YAMLImporter
from campus_nav.interfaces.ingestion_interface import (
    CampusDataImporterInterface,
    ImportedCampusData,
    ImportedEdge,
    ImportedNode,
)
from campus_nav.interfaces.rag_interface import (
    KnowledgeCategory,
    KnowledgeDocument,
    RAGRepositoryInterface,
)
from campus_nav.navigation.graph.geometry import haversine_distance_meters
from campus_nav.navigation.graph.models import CampusGraph, Edge, Node, NodeType, PathType

logger = get_logger(__name__)


def default_importers() -> list[CampusDataImporterInterface]:
    """The full set of importers this project ships. OSM XML/PBF and PDF
    are included so unsupported files get a clear, actionable error
    (see their modules) rather than being silently skipped or crashing
    the whole directory scan."""
    return [
        GeoJSONImporter(),
        CSVImporter(),
        YAMLImporter(),
        MarkdownImporter(),
        OSMXMLImporter(),
        OSMPBFImporter(),
        PDFImporter(),
    ]


class CampusDataIngestionPipeline:
    """Scans a directory of campus data files and merges the result into
    an existing CampusGraph and RAGRepositoryInterface."""

    def __init__(self, importers: list[CampusDataImporterInterface] | None = None) -> None:
        self._importers = importers if importers is not None else default_importers()

    def import_directory(self, directory: str | Path) -> ImportedCampusData:
        """Scan `directory` (non-recursively) and import every file a
        registered importer claims. Files with no matching importer are
        skipped with a logged warning, not an error -- an unrelated file
        sitting in the data directory (e.g. a README) is normal, not a
        pipeline failure. A file whose importer *does* claim it but then
        fails to parse it IS logged as a warning and skipped, per
        IngestionError's recoverable-by-design contract -- one malformed
        file must not abort the entire ingestion run.
        """
        directory = Path(directory)
        combined = ImportedCampusData()

        if not directory.exists() or not directory.is_dir():
            logger.warning(
                "Campus data directory not found; skipping ingestion",
                extra={"directory": str(directory)},
            )
            return combined

        for file_path in sorted(directory.iterdir()):
            if not file_path.is_file():
                continue

            importer = self._find_importer(file_path)
            if importer is None:
                logger.debug(
                    "No importer registered for file; skipping", extra={"file": str(file_path)}
                )
                continue

            try:
                data = importer.import_data(file_path)
                combined.merge(data)
                logger.info(
                    "Imported campus data file",
                    extra={
                        "file": str(file_path),
                        "nodes": len(data.nodes),
                        "edges": len(data.edges),
                        "knowledge_entries": len(data.knowledge_entries),
                    },
                )
            except IngestionError as exc:
                logger.warning(
                    "Skipping file that failed to import",
                    extra={"file": str(file_path), "error": str(exc)},
                )

        return combined

    def _find_importer(self, path: Path) -> CampusDataImporterInterface | None:
        for importer in self._importers:
            if importer.can_handle(path):
                return importer
        return None

    def apply_to_graph(self, data: ImportedCampusData, graph: CampusGraph) -> None:
        """Merge imported nodes/edges into `graph` using its existing,
        unmodified `add_node`/`add_edge` API. Nodes/edges whose id already
        exists in `graph` are skipped (logged), not overwritten -- the
        base pilot graph (or a previous import) always wins on conflict,
        since silently replacing a node an active route might reference
        is a correctness risk this pipeline should not take unsupervised.
        """
        for imported_node in data.nodes:
            if imported_node.id in graph.nodes:
                logger.warning(
                    "Skipping duplicate node id during import", extra={"node_id": imported_node.id}
                )
                continue
            graph.add_node(_to_domain_node(imported_node))

        for imported_edge in data.edges:
            if imported_edge.id in graph.edges:
                logger.warning(
                    "Skipping duplicate edge id during import", extra={"edge_id": imported_edge.id}
                )
                continue
            if (
                imported_edge.from_node_id not in graph.nodes
                or imported_edge.to_node_id not in graph.nodes
            ):
                logger.warning(
                    "Skipping edge referencing unknown node(s)",
                    extra={"edge_id": imported_edge.id},
                )
                continue

            distance_meters = imported_edge.distance_meters
            if distance_meters is None or distance_meters <= 0:
                from_node = graph.get_node(imported_edge.from_node_id)
                to_node = graph.get_node(imported_edge.to_node_id)
                distance_meters = haversine_distance_meters(
                    from_node.latitude, from_node.longitude, to_node.latitude, to_node.longitude
                )

            if distance_meters <= 0:
                logger.warning(
                    "Skipping edge with non-positive distance (identical endpoint coordinates)",
                    extra={"edge_id": imported_edge.id},
                )
                continue

            graph.add_edge(_to_domain_edge(imported_edge, distance_meters))

    def apply_to_knowledge_base(
        self, data: ImportedCampusData, knowledge_base: RAGRepositoryInterface
    ) -> None:
        """Merge imported knowledge entries into `knowledge_base` using
        its existing, unmodified `add_document` API."""
        for entry in data.knowledge_entries:
            try:
                category = KnowledgeCategory(entry.category)
            except ValueError:
                category = KnowledgeCategory.GENERAL
            knowledge_base.add_document(
                KnowledgeDocument(
                    id=entry.id,
                    title=entry.title,
                    content=entry.content,
                    category=category,
                    metadata=entry.metadata,
                    source=entry.metadata.get("source_file", "ingestion_pipeline"),
                )
            )


def _to_domain_node(imported: ImportedNode) -> Node:
    try:
        node_type = NodeType(imported.node_type)
    except ValueError:
        node_type = NodeType.WAYPOINT

    description = imported.landmark_description
    if imported.aliases and description:
        description = f"{description} Also known as: {', '.join(imported.aliases)}."

    return Node(
        id=imported.id,
        name=imported.name,
        latitude=imported.latitude,
        longitude=imported.longitude,
        node_type=node_type,
        floor=imported.floor,
        is_accessible=imported.is_accessible,
        landmark_description=description,
    )


def _to_domain_edge(imported: ImportedEdge, distance_meters: float) -> Edge:
    try:
        path_type = PathType(imported.path_type)
    except ValueError:
        path_type = PathType.OUTDOOR_PATH

    return Edge(
        id=imported.id,
        from_node_id=imported.from_node_id,
        to_node_id=imported.to_node_id,
        distance_meters=distance_meters,
        path_type=path_type,
        is_accessible=imported.is_accessible,
        bidirectional=imported.bidirectional,
        landmark_description=imported.landmark_description,
    )
