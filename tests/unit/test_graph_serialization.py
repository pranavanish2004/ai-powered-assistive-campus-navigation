"""Tests for navigation.graph.serialization (JSON file I/O boundary)."""

import json
import tempfile
from pathlib import Path

import pytest

from campus_nav.core.exceptions import GraphSerializationError
from campus_nav.navigation.graph.serialization import (
    load_graph_json,
    loads_graph_json,
    save_graph_json,
)

pytestmark = pytest.mark.unit


def test_save_and_load_round_trip(simple_graph):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "graph.json"
        save_graph_json(simple_graph, path)
        loaded = load_graph_json(path)
        assert loaded.node_count() == simple_graph.node_count()
        assert loaded.edge_count() == simple_graph.edge_count()


def test_save_creates_missing_parent_directories(simple_graph):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "nested" / "dir" / "graph.json"
        save_graph_json(simple_graph, path)
        assert path.exists()


def test_load_nonexistent_file_raises_graph_serialization_error():
    with pytest.raises(GraphSerializationError):
        load_graph_json("/nonexistent/path/graph.json")


def test_load_malformed_json_raises_graph_serialization_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(GraphSerializationError):
            load_graph_json(path)


def test_load_json_with_dangling_edge_reference_raises_validation_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bad_graph.json"
        data = {
            "nodes": [{"id": "A", "name": "A", "latitude": 0, "longitude": 0}],
            "edges": [
                {
                    "id": "AB",
                    "from_node_id": "A",
                    "to_node_id": "does-not-exist",
                    "distance_meters": 5,
                }
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        from campus_nav.core.exceptions import GraphValidationError

        with pytest.raises(GraphValidationError):
            load_graph_json(path)


def test_loads_graph_json_from_string(simple_graph):
    raw = json.dumps(simple_graph.to_dict())
    restored = loads_graph_json(raw)
    assert restored.node_count() == simple_graph.node_count()


def test_loads_graph_json_malformed_string_raises():
    with pytest.raises(GraphSerializationError):
        loads_graph_json("{not json")


def test_pilot_graph_fixture_loads_and_validates(pilot_graph):
    assert pilot_graph.node_count() == 8
    assert pilot_graph.edge_count() == 10
    assert pilot_graph.validate() == []
