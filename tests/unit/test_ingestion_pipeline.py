"""Tests for CampusDataIngestionPipeline: directory scanning, per-file
error isolation, and merging imported data into an existing CampusGraph
and RAGRepositoryInterface without modifying either class."""

from pathlib import Path

import pytest

from campus_nav.infrastructure.ingestion.ingestion_pipeline import (
    CampusDataIngestionPipeline,
    default_importers,
)
from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.interfaces.ingestion_interface import ImportedCampusData, ImportedEdge, ImportedNode
from campus_nav.navigation.graph.models import CampusGraph

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path("tests/fixtures/ingestion")


def test_default_importers_includes_every_format():
    importers = default_importers()
    assert len(importers) == 7  # geojson, csv, yaml, markdown, osm xml, osm pbf, pdf


def test_import_directory_scans_and_merges_all_supported_files():
    pipeline = CampusDataIngestionPipeline()
    data = pipeline.import_directory(FIXTURE_DIR)

    node_ids = {n.id for n in data.nodes}
    # From sample.geojson, sample.csv, sample.yaml, sample_node.md
    assert "geo_node_a" in node_ids
    assert "csv_node_a" in node_ids
    assert "yaml_node_a" in node_ids
    assert "md_node_1" in node_ids


def test_import_directory_skips_malformed_file_without_aborting_others():
    pipeline = CampusDataIngestionPipeline()
    # malformed.geojson exists alongside valid files in FIXTURE_DIR --
    # the run must still succeed and import everything else.
    data = pipeline.import_directory(FIXTURE_DIR)
    assert len(data.nodes) > 0


def test_import_directory_missing_directory_returns_empty_data():
    pipeline = CampusDataIngestionPipeline()
    data = pipeline.import_directory("/nonexistent/directory")
    assert data.nodes == []
    assert data.edges == []
    assert data.knowledge_entries == []


def test_import_directory_ignores_unrecognized_files(tmp_path):
    (tmp_path / "README.txt").write_text("not a campus data file")
    pipeline = CampusDataIngestionPipeline()
    data = pipeline.import_directory(tmp_path)
    assert data.nodes == []


# --- apply_to_graph ---


def test_apply_to_graph_adds_imported_nodes_and_edges():
    graph = CampusGraph()
    data = ImportedCampusData(
        nodes=[
            ImportedNode(id="a", name="A", latitude=17.0, longitude=78.0),
            ImportedNode(id="b", name="B", latitude=17.001, longitude=78.001),
        ],
        edges=[ImportedEdge(id="ab", from_node_id="a", to_node_id="b", distance_meters=50.0)],
    )
    pipeline = CampusDataIngestionPipeline()
    pipeline.apply_to_graph(data, graph)

    assert graph.node_count() == 2
    assert graph.edge_count() == 1
    assert graph.get_edge("ab").distance_meters == 50.0


def test_apply_to_graph_skips_duplicate_node_id_without_raising():
    graph = CampusGraph()
    from campus_nav.navigation.graph.models import Node

    graph.add_node(Node(id="a", name="Existing A", latitude=0, longitude=0))

    data = ImportedCampusData(
        nodes=[ImportedNode(id="a", name="Imported A", latitude=1, longitude=1)]
    )
    pipeline = CampusDataIngestionPipeline()
    pipeline.apply_to_graph(data, graph)  # must not raise

    assert graph.get_node("a").name == "Existing A"  # original wins, not overwritten


def test_apply_to_graph_skips_edge_referencing_unknown_node():
    graph = CampusGraph()
    data = ImportedCampusData(
        edges=[
            ImportedEdge(
                id="e1", from_node_id="missing_a", to_node_id="missing_b", distance_meters=10.0
            )
        ]
    )
    pipeline = CampusDataIngestionPipeline()
    pipeline.apply_to_graph(data, graph)  # must not raise
    assert graph.edge_count() == 0


def test_apply_to_graph_computes_distance_when_none_provided():
    graph = CampusGraph()
    data = ImportedCampusData(
        nodes=[
            ImportedNode(id="a", name="A", latitude=17.0, longitude=78.0),
            ImportedNode(id="b", name="B", latitude=17.001, longitude=78.001),
        ],
        edges=[ImportedEdge(id="ab", from_node_id="a", to_node_id="b", distance_meters=None)],
    )
    pipeline = CampusDataIngestionPipeline()
    pipeline.apply_to_graph(data, graph)
    assert graph.get_edge("ab").distance_meters > 0  # computed via haversine, not defaulted to 0.0


def test_apply_to_graph_skips_edge_with_identical_endpoint_coordinates():
    """An edge with no explicit distance between two identical coordinates
    computes to exactly 0 -- CampusGraph rejects non-positive distances,
    so this must be skipped with a warning, not raise or corrupt the graph."""
    graph = CampusGraph()
    data = ImportedCampusData(
        nodes=[
            ImportedNode(id="a", name="A", latitude=17.0, longitude=78.0),
            ImportedNode(id="b", name="B", latitude=17.0, longitude=78.0),
        ],
        edges=[ImportedEdge(id="ab", from_node_id="a", to_node_id="b", distance_meters=None)],
    )
    pipeline = CampusDataIngestionPipeline()
    pipeline.apply_to_graph(data, graph)  # must not raise
    assert graph.edge_count() == 0


def test_apply_to_graph_full_directory_produces_valid_graph():
    graph = CampusGraph()
    pipeline = CampusDataIngestionPipeline()
    data = pipeline.import_directory(FIXTURE_DIR)
    pipeline.apply_to_graph(data, graph)
    assert graph.validate() == [] or all("isolated" in w for w in graph.validate())


# --- apply_to_knowledge_base ---


def test_apply_to_knowledge_base_adds_documents():
    kb = KeywordKnowledgeBase()
    pipeline = CampusDataIngestionPipeline()
    data = pipeline.import_directory(FIXTURE_DIR)
    pipeline.apply_to_knowledge_base(data, kb)

    assert kb.document_count() == len(data.knowledge_entries)
    results = kb.search("standalone fact")
    assert any(r.document.id == "yaml_fact_1" for r in results)


def test_apply_to_knowledge_base_unknown_category_falls_back_to_general():
    from campus_nav.interfaces.ingestion_interface import ImportedKnowledgeEntry

    kb = KeywordKnowledgeBase()
    data = ImportedCampusData(
        knowledge_entries=[
            ImportedKnowledgeEntry(
                id="x", title="X", content="content", category="not-a-real-category"
            )
        ]
    )
    pipeline = CampusDataIngestionPipeline()
    pipeline.apply_to_knowledge_base(data, kb)  # must not raise
    assert kb.document_count() == 1
