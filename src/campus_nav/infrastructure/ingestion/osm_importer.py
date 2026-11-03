"""
OpenStreetMap raw-format importers.

Honest scoping note
---------------------
Raw OSM XML and PBF parsing requires heavy, compiled bindings (`osmium`,
`pyosmium`, or `imposm.parser`) with platform-specific build requirements
-- exactly the class of dependency this project has consistently avoided
making a hard requirement (see Phase 1.1's rationale for moving
`openai-whisper` to an optional extra, and Phase 1.3's for YOLO/EasyOCR/
MiDaS). Rather than bundle a heavy, fragile binary dependency for a format
conversion that standard external tools already do well, this module
raises a clear, actionable error pointing at the conversion step instead
of silently failing or producing incorrect data.

The recommended workflow for real OpenStreetMap data:

    osmium export campus.osm.pbf -o campus.geojson
    # or: ogr2ogr -f GeoJSON campus.geojson campus.osm.xml

then import the resulting `campus.geojson` via `GeoJSONImporter`, which
*is* fully implemented (pure stdlib `json`, no heavy dependencies). This
mirrors how real-world OSM-to-application pipelines are typically built in
practice -- conversion is a one-time or occasional external step, not
something the application itself needs to re-implement.
"""

from __future__ import annotations

from pathlib import Path

from campus_nav.core.exceptions import IngestionError
from campus_nav.interfaces.ingestion_interface import (
    CampusDataImporterInterface,
    ImportedCampusData,
)

_CONVERSION_HINT = (
    "Raw OSM {fmt} parsing is not implemented -- this project avoids the heavy, "
    "compiled dependencies (osmium/pyosmium) that would require. Convert to "
    "GeoJSON first (e.g. `osmium export input.{ext} -o output.geojson`), then "
    "import the .geojson file with GeoJSONImporter."
)


class OSMXMLImporter(CampusDataImporterInterface):
    """Recognizes .osm/.xml OSM files but does not parse them directly --
    see module docstring for why, and the required conversion step."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in (".osm",)

    def import_data(self, path: Path) -> ImportedCampusData:
        raise IngestionError(_CONVERSION_HINT.format(fmt="XML", ext="osm"))


class OSMPBFImporter(CampusDataImporterInterface):
    """Recognizes .osm.pbf OSM files but does not parse them directly --
    see module docstring for why, and the required conversion step."""

    def can_handle(self, path: Path) -> bool:
        return path.name.lower().endswith(".osm.pbf")

    def import_data(self, path: Path) -> ImportedCampusData:
        raise IngestionError(_CONVERSION_HINT.format(fmt="PBF", ext="osm.pbf"))
