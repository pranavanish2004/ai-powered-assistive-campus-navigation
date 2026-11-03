"""Tests for the honest-scoping OSM XML/PBF and PDF stub importers --
each recognizes its file extension but raises a clear, actionable
IngestionError pointing at the required conversion/extraction step,
rather than silently failing or fabricating a parse."""

from pathlib import Path

import pytest

from campus_nav.core.exceptions import IngestionError
from campus_nav.infrastructure.ingestion.osm_importer import OSMPBFImporter, OSMXMLImporter
from campus_nav.infrastructure.ingestion.pdf_importer import PDFImporter

pytestmark = pytest.mark.unit


def test_osm_xml_importer_recognizes_extension():
    importer = OSMXMLImporter()
    assert importer.can_handle(Path("campus.osm")) is True
    assert importer.can_handle(Path("campus.geojson")) is False


def test_osm_xml_importer_raises_actionable_error():
    importer = OSMXMLImporter()
    with pytest.raises(IngestionError, match="GeoJSONImporter"):
        importer.import_data(Path("campus.osm"))


def test_osm_pbf_importer_recognizes_extension():
    importer = OSMPBFImporter()
    assert importer.can_handle(Path("campus.osm.pbf")) is True
    assert importer.can_handle(Path("campus.osm")) is False


def test_osm_pbf_importer_raises_actionable_error():
    importer = OSMPBFImporter()
    with pytest.raises(IngestionError, match="GeoJSONImporter"):
        importer.import_data(Path("campus.osm.pbf"))


def test_pdf_importer_recognizes_extension():
    importer = PDFImporter()
    assert importer.can_handle(Path("brochure.pdf")) is True
    assert importer.can_handle(Path("brochure.docx")) is False


def test_pdf_importer_raises_actionable_error():
    importer = PDFImporter()
    with pytest.raises(IngestionError, match="not yet implemented"):
        importer.import_data(Path("brochure.pdf"))
