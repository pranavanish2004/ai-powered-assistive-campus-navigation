"""
GeoJSON importer.

This is the primary integration point for OpenStreetMap data. Raw OSM
XML/PBF parsing requires heavy, compiled bindings (`osmium`, `pyosmium`)
that would break this project's established "no heavy compiled
dependencies by default" pattern (see Phase 1.1's rationale for moving
Whisper out of the default install). The standard, widely-used workaround
is exactly what this importer expects as input: OSM data pre-converted to
GeoJSON via `osmium export`, `ogr2ogr`, or the Overpass API's own GeoJSON
output (`overpass-turbo.eu` can export directly). See
`osm_importer.py` for the explicit stub documenting this decision and
pointing at the conversion step.

Expected structure
-------------------
A standard GeoJSON `FeatureCollection`:

- `Point` features become `ImportedNode`s. Recognized `properties`:
  `id` (required), `name` (required), `node_type`, `floor`,
  `is_accessible`, `landmark_description`, `aliases` (list of strings).
- `LineString` features become `ImportedEdge`s. Since a raw OSM way has
  no concept of "which imported node is each endpoint", this importer
  requires explicit `from_node_id`/`to_node_id` properties on each
  LineString feature (set during the OSM-to-GeoJSON conversion step, or
  by hand for a small campus dataset) rather than attempting fragile
  coordinate-snapping. Recognized `properties`: `id`, `from_node_id`,
  `to_node_id` (all required), `distance_meters` (optional -- computed
  from endpoint coordinates via haversine if absent), `path_type`,
  `is_accessible`, `bidirectional`, `landmark_description`.
"""

from __future__ import annotations

import json
from pathlib import Path

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


class GeoJSONImporter(CampusDataImporterInterface):
    """Imports nodes (from Point features) and edges (from LineString
    features) from a GeoJSON FeatureCollection."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in (".geojson", ".json.geojson")

    def import_data(self, path: Path) -> ImportedCampusData:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestionError(f"Failed to read/parse GeoJSON file {path}: {exc}") from exc

        features = raw.get("features", [])
        if not isinstance(features, list):
            raise IngestionError(f"{path}: 'features' must be a list.")

        data = ImportedCampusData()
        # Nodes must be collected before edges so distance_meters can be
        # computed from endpoint coordinates when not explicitly provided.
        node_lookup: dict[str, ImportedNode] = {}

        for feature in features:
            geometry = feature.get("geometry", {})
            geometry_type = geometry.get("type")
            properties = feature.get("properties", {})

            if geometry_type == "Point":
                node = self._parse_point_feature(geometry, properties, path)
                if node is not None:
                    data.nodes.append(node)
                    node_lookup[node.id] = node
                    if node.landmark_description:
                        data.knowledge_entries.append(
                            ImportedKnowledgeEntry(
                                id=f"geojson_{node.id}",
                                title=node.name,
                                content=self._content_with_aliases(node),
                                category=node.node_type,
                                metadata={"node_id": node.id, "source_file": str(path)},
                            )
                        )

        for feature in features:
            geometry = feature.get("geometry", {})
            if geometry.get("type") == "LineString":
                edge = self._parse_linestring_feature(
                    geometry, feature.get("properties", {}), node_lookup, path
                )
                if edge is not None:
                    data.edges.append(edge)

        return data

    @staticmethod
    def _content_with_aliases(node: ImportedNode) -> str:
        base = node.landmark_description or ""
        if node.aliases:
            return f"{base} Also known as: {', '.join(node.aliases)}."
        return base

    def _parse_point_feature(
        self, geometry: dict, properties: dict, path: Path
    ) -> ImportedNode | None:
        coordinates = geometry.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            logger.warning(
                "Skipping Point feature with invalid coordinates", extra={"path": str(path)}
            )
            return None

        node_id = properties.get("id")
        name = properties.get("name")
        if not node_id or not name:
            logger.warning(
                "Skipping Point feature missing required 'id'/'name' properties",
                extra={"path": str(path), "properties": str(properties)[:200]},
            )
            return None

        longitude, latitude = coordinates[0], coordinates[1]  # GeoJSON order is [lon, lat]
        return ImportedNode(
            id=str(node_id),
            name=str(name),
            latitude=float(latitude),
            longitude=float(longitude),
            node_type=str(properties.get("node_type", "waypoint")),
            floor=int(properties.get("floor", 0)),
            is_accessible=bool(properties.get("is_accessible", True)),
            landmark_description=properties.get("landmark_description"),
            aliases=list(properties.get("aliases", [])),
        )

    def _parse_linestring_feature(
        self, geometry: dict, properties: dict, node_lookup: dict[str, ImportedNode], path: Path
    ) -> ImportedEdge | None:
        edge_id = properties.get("id")
        from_node_id = properties.get("from_node_id")
        to_node_id = properties.get("to_node_id")

        if not edge_id or not from_node_id or not to_node_id:
            logger.warning(
                "Skipping LineString feature missing required id/from_node_id/to_node_id",
                extra={"path": str(path), "properties": str(properties)[:200]},
            )
            return None

        distance_meters = properties.get("distance_meters")
        if distance_meters is None:
            from_node = node_lookup.get(str(from_node_id))
            to_node = node_lookup.get(str(to_node_id))
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
            id=str(edge_id),
            from_node_id=str(from_node_id),
            to_node_id=str(to_node_id),
            distance_meters=float(distance_meters),
            path_type=str(properties.get("path_type", "outdoor_path")),
            is_accessible=bool(properties.get("is_accessible", True)),
            bidirectional=bool(properties.get("bidirectional", True)),
            landmark_description=properties.get("landmark_description"),
        )
