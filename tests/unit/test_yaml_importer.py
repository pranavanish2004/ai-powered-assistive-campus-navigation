"""Tests for YAMLImporter."""

from pathlib import Path

import pytest

from campus_nav.core.exceptions import IngestionError
from campus_nav.infrastructure.ingestion.yaml_importer import YAMLImporter

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path("tests/fixtures/ingestion")


def test_can_handle_yaml_extensions():
    importer = YAMLImporter()
    assert importer.can_handle(Path("data.yaml")) is True
    assert importer.can_handle(Path("data.yml")) is True
    assert importer.can_handle(Path("data.csv")) is False


def test_imports_nodes():
    importer = YAMLImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.yaml")
    node_ids = {n.id for n in data.nodes}
    assert node_ids == {"yaml_node_a", "yaml_node_b"}


def test_imports_edges_with_computed_distance():
    importer = YAMLImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.yaml")
    assert len(data.edges) == 1
    edge = data.edges[0]
    assert edge.from_node_id == "yaml_node_a"
    assert edge.to_node_id == "yaml_node_b"
    assert edge.distance_meters > 0


def test_imports_standalone_knowledge_entries():
    importer = YAMLImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.yaml")
    assert any(e.id == "yaml_fact_1" for e in data.knowledge_entries)


def test_file_with_only_knowledge_section(tmp_path):
    yaml_file = tmp_path / "knowledge_only.yaml"
    yaml_file.write_text("knowledge:\n  - id: k1\n    title: Fact\n    content: Just a fact.\n")
    importer = YAMLImporter()
    data = importer.import_data(yaml_file)
    assert data.nodes == []
    assert data.edges == []
    assert len(data.knowledge_entries) == 1


def test_edge_with_unresolvable_endpoints_and_no_distance_is_skipped(tmp_path):
    yaml_file = tmp_path / "bad_edge.yaml"
    yaml_file.write_text(
        "edges:\n  - id: e1\n    from_node_id: missing_a\n    to_node_id: missing_b\n"
    )
    importer = YAMLImporter()
    data = importer.import_data(yaml_file)
    assert data.edges == []


def test_malformed_yaml_raises_ingestion_error(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("nodes: [unterminated")
    importer = YAMLImporter()
    with pytest.raises(IngestionError):
        importer.import_data(yaml_file)


def test_non_mapping_top_level_raises_ingestion_error(tmp_path):
    yaml_file = tmp_path / "list.yaml"
    yaml_file.write_text("- just\n- a\n- list\n")
    importer = YAMLImporter()
    with pytest.raises(IngestionError):
        importer.import_data(yaml_file)


def test_missing_file_raises_ingestion_error():
    importer = YAMLImporter()
    with pytest.raises(IngestionError):
        importer.import_data(Path("/nonexistent/file.yaml"))
