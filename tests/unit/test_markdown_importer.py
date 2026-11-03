"""Tests for MarkdownImporter."""

from pathlib import Path

import pytest

from campus_nav.core.exceptions import IngestionError
from campus_nav.infrastructure.ingestion.markdown_importer import MarkdownImporter

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path("tests/fixtures/ingestion")


def test_can_handle_markdown_extensions():
    importer = MarkdownImporter()
    assert importer.can_handle(Path("note.md")) is True
    assert importer.can_handle(Path("note.markdown")) is True
    assert importer.can_handle(Path("note.txt")) is False


def test_note_without_coordinates_produces_knowledge_entry_only():
    importer = MarkdownImporter()
    data = importer.import_data(FIXTURE_DIR / "sample_note.md")
    assert len(data.knowledge_entries) == 1
    assert data.nodes == []
    entry = data.knowledge_entries[0]
    assert entry.id == "md_note_1"
    assert "ramp location" in entry.content
    assert entry.metadata["node_id"] == "yaml_node_a"


def test_note_with_coordinates_produces_node_and_knowledge_entry():
    importer = MarkdownImporter()
    data = importer.import_data(FIXTURE_DIR / "sample_node.md")
    assert len(data.nodes) == 1
    assert len(data.knowledge_entries) == 1
    node = data.nodes[0]
    assert node.id == "md_node_1"
    assert node.latitude == 17.65
    assert node.longitude == 78.25


def test_missing_frontmatter_raises_ingestion_error(tmp_path):
    md_file = tmp_path / "no_frontmatter.md"
    md_file.write_text("Just plain text, no frontmatter block.")
    importer = MarkdownImporter()
    with pytest.raises(IngestionError):
        importer.import_data(md_file)


def test_frontmatter_missing_required_fields_raises_ingestion_error(tmp_path):
    md_file = tmp_path / "incomplete.md"
    md_file.write_text("---\ncategory: general\n---\nBody text.")
    importer = MarkdownImporter()
    with pytest.raises(IngestionError):
        importer.import_data(md_file)


def test_malformed_frontmatter_yaml_raises_ingestion_error(tmp_path):
    md_file = tmp_path / "bad_yaml.md"
    md_file.write_text("---\nid: [unterminated\n---\nBody.")
    importer = MarkdownImporter()
    with pytest.raises(IngestionError):
        importer.import_data(md_file)


def test_invalid_coordinates_falls_back_to_knowledge_entry_only(tmp_path):
    md_file = tmp_path / "bad_coords.md"
    md_file.write_text(
        "---\nid: x\ntitle: X\nlatitude: not-a-number\nlongitude: 78.0\n---\nBody text."
    )
    importer = MarkdownImporter()
    data = importer.import_data(md_file)
    assert len(data.knowledge_entries) == 1
    assert data.nodes == []


def test_missing_file_raises_ingestion_error():
    importer = MarkdownImporter()
    with pytest.raises(IngestionError):
        importer.import_data(Path("/nonexistent/file.md"))
