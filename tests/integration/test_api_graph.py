"""Integration tests for the /graph API endpoints."""

import pytest
from fastapi.testclient import TestClient

from campus_nav.api.main import create_app

pytestmark = pytest.mark.integration

FIXTURE_GRAPH = "tests/fixtures/sample_campus_graph.json"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app(graph_path=FIXTURE_GRAPH)
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_full_graph_returns_all_nodes_and_edges(client):
    response = client.get("/graph")
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) == 8
    assert len(data["edges"]) == 10


def test_get_single_node(client):
    response = client.get("/graph/nodes/main_gate")
    assert response.status_code == 200
    assert response.json()["name"] == "Main Gate"


def test_get_nonexistent_node_returns_404(client):
    response = client.get("/graph/nodes/does-not-exist")
    assert response.status_code == 404


def test_get_node_neighbors(client):
    response = client.get("/graph/nodes/main_gate/neighbors")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_get_neighbors_of_nonexistent_node_returns_404(client):
    response = client.get("/graph/nodes/does-not-exist/neighbors")
    assert response.status_code == 404
