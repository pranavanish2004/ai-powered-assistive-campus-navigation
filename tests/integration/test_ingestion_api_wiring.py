"""Integration test: verifies Settings.campus_data_directory actually gets
scanned at app startup and its nodes/edges/knowledge merged into the live
graph and knowledge base -- exercising create_app()'s real wiring, not
just the pipeline components in isolation."""

import pytest
from fastapi.testclient import TestClient

from campus_nav.api.main import create_app
from campus_nav.config.settings import get_settings

pytestmark = pytest.mark.integration

FIXTURE_GRAPH = "tests/fixtures/sample_campus_graph.json"
INGESTION_FIXTURE_DIR = "tests/fixtures/ingestion"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("CAMPUS_DATA_DIRECTORY", INGESTION_FIXTURE_DIR)
    get_settings.cache_clear()  # force re-reading the env vars just set
    app = create_app(graph_path=FIXTURE_GRAPH)
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()  # don't leak this config into later tests


def test_imported_nodes_appear_in_graph_endpoint(client):
    response = client.get("/graph")
    assert response.status_code == 200
    node_ids = {n["id"] for n in response.json()["nodes"]}
    # Base pilot graph nodes are still present...
    assert "main_gate" in node_ids
    # ...alongside nodes merged in from the ingestion fixture directory.
    assert "geo_node_a" in node_ids
    assert "csv_node_a" in node_ids
    assert "yaml_node_a" in node_ids
    assert "md_node_1" in node_ids


def test_imported_knowledge_is_searchable_via_assistant(client):
    response = client.post("/assistant/ask", json={"query": "standalone fact"})
    assert response.status_code == 200
    assert response.json()["used_rag"] is True


def test_base_graph_navigation_still_works_with_ingestion_enabled(client):
    """Regression guard: enabling ingestion must not disturb the base
    pilot graph's own routing behavior."""
    response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "library"},
    )
    assert response.status_code == 201


def test_no_ingestion_directory_configured_leaves_graph_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test2.db")
    monkeypatch.delenv("CAMPUS_DATA_DIRECTORY", raising=False)
    get_settings.cache_clear()
    app = create_app(graph_path=FIXTURE_GRAPH)
    with TestClient(app) as test_client:
        response = test_client.get("/graph")
        node_ids = {n["id"] for n in response.json()["nodes"]}
        assert "geo_node_a" not in node_ids
        assert len(node_ids) == 8  # exactly the base pilot fixture graph's node count
    get_settings.cache_clear()
