"""Tests for CSVImporter."""

from pathlib import Path

import pytest

from campus_nav.core.exceptions import IngestionError
from campus_nav.infrastructure.ingestion.csv_importer import CSVImporter

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path("tests/fixtures/ingestion")


def test_can_handle_csv_extension():
    importer = CSVImporter()
    assert importer.can_handle(Path("data.csv")) is True
    assert importer.can_handle(Path("data.yaml")) is False


def test_imports_rows_as_nodes():
    importer = CSVImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.csv")
    node_ids = {n.id for n in data.nodes}
    assert node_ids == {"csv_node_a", "csv_node_b"}


def test_is_accessible_defaults_true_when_blank():
    importer = CSVImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.csv")
    node_b = next(n for n in data.nodes if n.id == "csv_node_b")
    assert node_b.is_accessible is False  # explicitly "false" in the fixture


def test_aliases_parsed_from_semicolon_list():
    importer = CSVImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.csv")
    node_a = next(n for n in data.nodes if n.id == "csv_node_a")
    assert node_a.aliases == ["Node CSV A"]


def test_node_with_description_produces_knowledge_entry():
    importer = CSVImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.csv")
    assert any(e.id == "csv_csv_node_a" for e in data.knowledge_entries)


def test_node_without_description_produces_no_knowledge_entry():
    importer = CSVImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.csv")
    assert not any(e.id == "csv_csv_node_b" for e in data.knowledge_entries)


def test_missing_required_column_raises_ingestion_error(tmp_path):
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("id,name\nx,Missing Coordinates\n")
    importer = CSVImporter()
    with pytest.raises(IngestionError):
        importer.import_data(bad_file)


def test_malformed_row_is_skipped_not_raised(tmp_path):
    csv_file = tmp_path / "partial.csv"
    csv_file.write_text(
        "id,name,latitude,longitude\n"
        "good,Good Node,17.0,78.0\n"
        "bad,Bad Node,not-a-number,78.0\n"
    )
    importer = CSVImporter()
    data = importer.import_data(csv_file)
    assert len(data.nodes) == 1
    assert data.nodes[0].id == "good"


def test_missing_file_raises_ingestion_error():
    importer = CSVImporter()
    with pytest.raises(IngestionError):
        importer.import_data(Path("/nonexistent/file.csv"))
