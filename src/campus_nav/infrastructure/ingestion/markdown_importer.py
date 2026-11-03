"""
Markdown importer -- one landmark or fact per file, in the common
"frontmatter + prose" convention (YAML metadata block delimited by `---`,
followed by free-form Markdown content). Well-suited to exactly the kind
of personal, narrative campus notes described for future survey data:
accessibility notes, landmark write-ups, one file per location.

Expected format:

```markdown
---
id: library_entrance_note
title: Central Library Accessible Entrance
category: accessibility
node_id: library      # optional: links this note to an existing graph node
latitude: 17.5948      # optional: only if this note ALSO describes a new node
longitude: 78.1245     # optional, same as above
---
The accessible entrance is on the north side of the building, past the
bicycle racks. There is a smooth, gently-sloped ramp with a handrail on
the left side.
```

If `latitude`/`longitude` are present in the frontmatter, an `ImportedNode`
is produced in addition to the knowledge entry (the file describes a new
location, not just a note about an existing one). If absent, only a
knowledge entry is produced -- most personal notes will be about a
location already in the graph, referenced via `node_id` for traceability.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from campus_nav.core.exceptions import IngestionError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.ingestion_interface import (
    CampusDataImporterInterface,
    ImportedCampusData,
    ImportedKnowledgeEntry,
    ImportedNode,
)

logger = get_logger(__name__)

_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class MarkdownImporter(CampusDataImporterInterface):
    """Imports one knowledge entry (and optionally one node) per Markdown
    file with a YAML frontmatter block."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in (".md", ".markdown")

    def import_data(self, path: Path) -> ImportedCampusData:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(f"Failed to read Markdown file {path}: {exc}") from exc

        match = _FRONTMATTER_PATTERN.match(raw_text)
        if not match:
            raise IngestionError(f"{path}: expected a YAML frontmatter block delimited by '---'.")

        frontmatter_text, body = match.group(1), match.group(2).strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as exc:
            raise IngestionError(f"{path}: malformed YAML frontmatter: {exc}") from exc

        if (
            not isinstance(frontmatter, dict)
            or "id" not in frontmatter
            or "title" not in frontmatter
        ):
            raise IngestionError(
                f"{path}: frontmatter must be a mapping with at least 'id' and 'title'."
            )

        data = ImportedCampusData()
        data.knowledge_entries.append(
            ImportedKnowledgeEntry(
                id=str(frontmatter["id"]),
                title=str(frontmatter["title"]),
                content=body,
                category=str(frontmatter.get("category", "general")),
                metadata={
                    "source_file": str(path),
                    **({"node_id": frontmatter["node_id"]} if "node_id" in frontmatter else {}),
                },
            )
        )

        if "latitude" in frontmatter and "longitude" in frontmatter:
            try:
                data.nodes.append(
                    ImportedNode(
                        id=str(frontmatter.get("node_id", frontmatter["id"])),
                        name=str(frontmatter["title"]),
                        latitude=float(frontmatter["latitude"]),
                        longitude=float(frontmatter["longitude"]),
                        node_type=str(frontmatter.get("node_type", "landmark")),
                        is_accessible=bool(frontmatter.get("is_accessible", True)),
                        landmark_description=body,
                        aliases=list(frontmatter.get("aliases", [])),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Frontmatter had latitude/longitude but they were invalid; "
                    "importing the knowledge entry only",
                    extra={"path": str(path), "error": str(exc)},
                )

        return data
