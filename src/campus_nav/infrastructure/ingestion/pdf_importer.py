"""
PDF importer -- explicitly deferred future extension point.

The specification names PDF ingestion as future scope, not current scope.
This stub exists so the extension point is visible and registered in the
pipeline today (a `.pdf` file dropped into the data directory gets a
clear, actionable error rather than being silently ignored or crashing
the whole ingestion run), without speculatively implementing PDF
text/layout extraction against a format this project has no current
concrete use case for. When needed, this project already has a `pdf`
skill/dependency pattern established (see the existing PDF-handling
conventions elsewhere in the toolchain) to build against.
"""

from __future__ import annotations

from pathlib import Path

from campus_nav.core.exceptions import IngestionError
from campus_nav.interfaces.ingestion_interface import (
    CampusDataImporterInterface,
    ImportedCampusData,
)


class PDFImporter(CampusDataImporterInterface):
    """Recognizes .pdf files but does not parse them -- deferred future
    scope per the project specification."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def import_data(self, path: Path) -> ImportedCampusData:
        raise IngestionError(
            f"{path}: PDF ingestion is not yet implemented (explicitly deferred future "
            "scope). Extract the relevant text/data into a .yaml, .csv, or .md file first."
        )
