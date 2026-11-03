"""Tests for GeoJSONImporter."""

from pathlib import Path

import pytest

from campus_nav.core.exceptions import IngestionError
from campus_nav.infrastructure.ingestion.geojson_importer import GeoJSONImporter

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path("tests/fixtures/ingestion")


def test_can_handle_geojson_extension():
    importer = GeoJSONImporter()
    assert importer.can_handle(Path("map.geojson")) is True
    assert importer.can_handle(Path("map.csv")) is False


def test_imports_point_features_as_nodes():
    importer = GeoJSONImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.geojson")
    node_ids = {n.id for n in data.nodes}
    assert "geo_node_a" in node_ids
    assert "geo_node_b" in node_ids


def test_point_missing_required_fields_is_skipped_not_raised():
    importer = GeoJSONImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.geojson")
    # The fourth feature (missing "id") must be skipped, not crash the import.
    assert len(data.nodes) == 2


def test_node_with_landmark_description_produces_knowledge_entry():
    importer = GeoJSONImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.geojson")
    entry = next(e for e in data.knowledge_entries if "geo_node_a" in e.id)
    assert "First imported node." in entry.content
    assert "Node A" in entry.content  # alias woven into content


def test_imports_linestring_as_edge_with_explicit_from_to():
    importer = GeoJSONImporter()
    data = importer.import_data(FIXTURE_DIR / "sample.geojson")
    assert len(data.edges) == 1
    edge = data.edges[0]
    assert edge.from_node_id == "geo_node_a"
    assert edge.to_node_id == "geo_node_b"
    assert edge.distance_meters is not None
    assert edge.distance_meters > 0  # computed via haversine, no explicit value given


def test_malformed_geojson_raises_ingestion_error():
    importer = GeoJSONImporter()
    with pytest.raises(IngestionError):
        importer.import_data(FIXTURE_DIR / "malformed.geojson")


def test_missing_file_raises_ingestion_error():
    importer = GeoJSONImporter()
    with pytest.raises(IngestionError):
        importer.import_data(Path("/nonexistent/file.geojson"))


def test_features_key_not_a_list_raises_ingestion_error(tmp_path):
    import json

    bad_file = tmp_path / "bad.geojson"
    bad_file.write_text(json.dumps({"type": "FeatureCollection", "features": "not-a-list"}))
    importer = GeoJSONImporter()
    with pytest.raises(IngestionError):
        importer.import_data(bad_file)
