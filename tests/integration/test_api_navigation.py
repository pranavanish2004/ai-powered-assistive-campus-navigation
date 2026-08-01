"""Integration tests for the /navigation API endpoints -- full session
lifecycle through HTTP, including error-path status codes."""

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


def test_start_navigation_returns_route_and_session_id(client):
    response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "biomedical"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "navigation_session_id" in data
    assert data["route"]["node_ids"][0] == "main_gate"
    assert data["route"]["node_ids"][-1] == "biomedical"
    assert data["first_instruction"] is not None


def test_start_navigation_with_unknown_node_returns_404(client):
    response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "nonexistent"},
    )
    assert response.status_code == 404


def test_start_navigation_creates_new_user_session_when_omitted(client):
    response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "library"},
    )
    assert response.status_code == 201


def test_start_navigation_with_unknown_user_session_returns_404(client):
    response = client.post(
        "/navigation/sessions",
        json={
            "user_session_id": "does-not-exist",
            "origin_node_id": "main_gate",
            "destination_node_id": "library",
        },
    )
    assert response.status_code == 404


def test_position_update_on_unknown_session_returns_404(client):
    response = client.post(
        "/navigation/sessions/does-not-exist/position",
        json={"latitude": 17.594, "longitude": 78.123, "accuracy_meters": 5.0},
    )
    assert response.status_code == 404


def test_full_lifecycle_start_position_pause_resume_stop(client):
    start_response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "library"},
    )
    session_id = start_response.json()["navigation_session_id"]

    position_response = client.post(
        f"/navigation/sessions/{session_id}/position",
        json={"latitude": 17.594808, "longitude": 78.123377, "accuracy_meters": 5.0},
    )
    assert position_response.status_code == 200
    assert position_response.json()["state"] in ("navigating", "uncertain", "rerouting")

    pause_response = client.post(f"/navigation/sessions/{session_id}/pause")
    assert pause_response.status_code == 200

    resume_response = client.post(f"/navigation/sessions/{session_id}/resume")
    assert resume_response.status_code == 200

    stop_response = client.post(f"/navigation/sessions/{session_id}/stop")
    assert stop_response.status_code == 200

    # After stop, the session is removed from the registry.
    status_response = client.get(f"/navigation/sessions/{session_id}/status")
    assert status_response.status_code == 404


def test_pause_twice_returns_conflict(client):
    start_response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "library"},
    )
    session_id = start_response.json()["navigation_session_id"]

    first_pause = client.post(f"/navigation/sessions/{session_id}/pause")
    assert first_pause.status_code == 200

    second_pause = client.post(f"/navigation/sessions/{session_id}/pause")
    assert second_pause.status_code == 409


def test_route_alternatives_endpoint(client):
    response = client.post(
        "/navigation/routes/alternatives",
        json={"origin_node_id": "main_gate", "destination_node_id": "sncc", "max_routes": 3},
    )
    assert response.status_code == 200
    routes = response.json()["routes"]
    assert len(routes) >= 1


def test_route_alternatives_with_unknown_node_returns_404(client):
    response = client.post(
        "/navigation/routes/alternatives",
        json={"origin_node_id": "main_gate", "destination_node_id": "nonexistent"},
    )
    assert response.status_code == 404


def test_accessible_route_request_via_api(client):
    response = client.post(
        "/navigation/sessions",
        json={
            "origin_node_id": "main_gate",
            "destination_node_id": "hostel_h1",
            "accessibility_mode": "strict",
        },
    )
    assert response.status_code == 201
    assert response.json()["route"]["uses_inaccessible_edge"] is False


def test_get_status_reflects_current_state(client):
    start_response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "library"},
    )
    session_id = start_response.json()["navigation_session_id"]

    status_response = client.get(f"/navigation/sessions/{session_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["state"] == "navigating"
