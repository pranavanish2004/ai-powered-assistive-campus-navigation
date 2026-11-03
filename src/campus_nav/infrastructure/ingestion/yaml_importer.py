"""
YAML importer -- the richest hand-authorable format, since YAML alone
(unlike CSV) can naturally express nodes, edges, and standalone knowledge
entries (facts with no specific coordinate, e.g. "campus rules") in one
file.

Expected top-level structure:

```yaml
nodes:
  - id: library
    name: Central Library
    latitude: 17.5948
    longitude: 78.1245
    node_type: building_entrance
    is_accessible: true
    landmark_description: "..."
    aliases: ["Lib", "Central Lib"]

edges:
  - id: e_admin_library
    from_node_id: admin_block
    to_node_id: library
    distance_meters: 80          # optional; computed if omitted and both endpoints are known
    path_type: outdoor_path
    is_accessible: true

knowledge:
  - id: campus_rules_navigation
    title: Campus Navigation Rules
    content: "..."
    category: rules
```

All three top-level keys are optional -- a file can contain only
`knowledge`, for instance.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from campus_nav.core.exceptions import IngestionError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.ingestion_interface import (
    CampusDataImporterInterface,
    ImportedCampusData,
    ImportedEdge,
    ImportedKnowledgeEntry,
    ImportedNode,
)
from campus_nav.navigation.graph.geometry import haversine_distance_meters

logger = get_logger(__name__)


class YAMLImporter(CampusDataImporterInterface):
    """Imports nodes, edges, and standalone knowledge entries from a
    structured YAML file."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in (".yaml", ".yml")

    def import_data(self, path: Path) -> ImportedCampusData:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise IngestionError(f"Failed to read/parse YAML file {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise IngestionError(f"{path}: top-level YAML content must be a mapping.")

        data = ImportedCampusData()
        node_lookup: dict[str, ImportedNode] = {}

        for entry in raw.get("nodes", []) or []:
            node = self._parse_node(entry, path)
            if node is not None:
                data.nodes.append(node)
                node_lookup[node.id] = node

        for entry in raw.get("edges", []) or []:
            edge = self._parse_edge(entry, node_lookup, path)
            if edge is not None:
                data.edges.append(edge)

        for entry in raw.get("knowledge", []) or []:
            knowledge = self._parse_knowledge(entry, path)
            if knowledge is not None:
                data.knowledge_entries.append(knowledge)

        return data

    def _parse_node(self, entry: dict, path: Path) -> ImportedNode | None:
        try:
            return ImportedNode(
                id=str(entry["id"]),
                name=str(entry["name"]),
                latitude=float(entry["latitude"]),
                longitude=float(entry["longitude"]),
                node_type=str(entry.get("node_type", "waypoint")),
                floor=int(entry.get("floor", 0)),
                is_accessible=bool(entry.get("is_accessible", True)),
                landmark_description=entry.get("landmark_description"),
                aliases=list(entry.get("aliases", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Skipping malformed YAML node entry", extra={"path": str(path), "error": str(exc)}
            )
            return None

    def _parse_edge(
        self, entry: dict, node_lookup: dict[str, ImportedNode], path: Path
    ) -> ImportedEdge | None:
        try:
            edge_id = str(entry["id"])
            from_node_id = str(entry["from_node_id"])
            to_node_id = str(entry["to_node_id"])
        except KeyError as exc:
            logger.warning(
                "Skipping malformed YAML edge entry (missing required field)",
                extra={"path": str(path), "error": str(exc)},
            )
            return None

        distance_meters = entry.get("distance_meters")
        if distance_meters is None:
            from_node = node_lookup.get(from_node_id)
            to_node = node_lookup.get(to_node_id)
            if from_node is None or to_node is None:
                logger.warning(
                    "Skipping edge with unresolvable endpoints and no explicit distance_meters",
                    extra={"path": str(path), "edge_id": edge_id},
                )
                return None
            distance_meters = haversine_distance_meters(
                from_node.latitude, from_node.longitude, to_node.latitude, to_node.longitude
            )

        return ImportedEdge(
            id=edge_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            distance_meters=float(distance_meters),
            path_type=str(entry.get("path_type", "outdoor_path")),
            is_accessible=bool(entry.get("is_accessible", True)),
            bidirectional=bool(entry.get("bidirectional", True)),
            landmark_description=entry.get("landmark_description"),
        )

    def _parse_knowledge(self, entry: dict, path: Path) -> ImportedKnowledgeEntry | None:
        try:
            return ImportedKnowledgeEntry(
                id=str(entry["id"]),
                title=str(entry["title"]),
                content=str(entry["content"]),
                category=str(entry.get("category", "general")),
                metadata=dict(entry.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Skipping malformed YAML knowledge entry",
                extra={"path": str(path), "error": str(exc)},
            )
            return None
