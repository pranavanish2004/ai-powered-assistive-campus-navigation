"""
CSV importer -- a simple location gazetteer format, well-suited to
spreadsheet-based campus surveys (e.g. walking around campus with a phone,
logging name/lat/lon/notes into a spreadsheet, exporting to CSV).

Expected columns (header row required): `id`, `name`, `latitude`,
`longitude`. Optional columns: `node_type`, `is_accessible` (true/false),
`landmark_description`, `aliases` (semicolon-separated), `category`
(defaults to `node_type` if omitted, for the resulting knowledge entry).

CSV has no native way to express edges (path connectivity) -- this
importer produces nodes and knowledge entries only. Combine with a
GeoJSON file's LineString edges, or a YAML file's explicit edge list, to
add connectivity for CSV-imported locations.
"""

from __future__ import annotations

import csv
from pathlib import Path

from campus_nav.core.exceptions import IngestionError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.ingestion_interface import (
    CampusDataImporterInterface,
    ImportedCampusData,
    ImportedKnowledgeEntry,
    ImportedNode,
)

logger = get_logger(__name__)

_REQUIRED_COLUMNS = {"id", "name", "latitude", "longitude"}


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("true", "1", "yes", "y")


class CSVImporter(CampusDataImporterInterface):
    """Imports a simple location gazetteer CSV into nodes + knowledge entries."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def import_data(self, path: Path) -> ImportedCampusData:
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise IngestionError(f"{path}: CSV file has no header row.")

                missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
                if missing:
                    raise IngestionError(f"{path}: missing required columns: {sorted(missing)}")

                rows = list(reader)
        except OSError as exc:
            raise IngestionError(f"Failed to read CSV file {path}: {exc}") from exc

        data = ImportedCampusData()
        for row_number, row in enumerate(rows, start=2):  # header is row 1
            node = self._parse_row(row, path, row_number)
            if node is None:
                continue
            data.nodes.append(node)
            if node.landmark_description:
                data.knowledge_entries.append(
                    ImportedKnowledgeEntry(
                        id=f"csv_{node.id}",
                        title=node.name,
                        content=self._content_with_aliases(node),
                        category=row.get("category") or node.node_type,
                        metadata={"node_id": node.id, "source_file": str(path)},
                    )
                )
        return data

    @staticmethod
    def _content_with_aliases(node: ImportedNode) -> str:
        base = node.landmark_description or ""
        if node.aliases:
            return f"{base} Also known as: {', '.join(node.aliases)}."
        return base

    def _parse_row(self, row: dict, path: Path, row_number: int) -> ImportedNode | None:
        try:
            node_id = row["id"].strip()
            name = row["name"].strip()
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (KeyError, ValueError, AttributeError) as exc:
            logger.warning(
                "Skipping malformed CSV row",
                extra={"path": str(path), "row_number": row_number, "error": str(exc)},
            )
            return None

        if not node_id or not name:
            logger.warning(
                "Skipping CSV row with empty id/name",
                extra={"path": str(path), "row_number": row_number},
            )
            return None

        aliases_raw = row.get("aliases", "") or ""
        aliases = [a.strip() for a in aliases_raw.split(";") if a.strip()]

        return ImportedNode(
            id=node_id,
            name=name,
            latitude=latitude,
            longitude=longitude,
            node_type=row.get("node_type", "waypoint") or "waypoint",
            is_accessible=_parse_bool(row.get("is_accessible"), default=True),
            landmark_description=row.get("landmark_description") or None,
            aliases=aliases,
        )
