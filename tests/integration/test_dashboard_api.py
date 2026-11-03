"""Integration tests for the /dashboard/* API endpoints, using an injected
MockDashboardRepository via create_app's dashboard_repository_factory
parameter -- no real network calls to the external IITH Dashboard service."""

import pytest
from fastapi.testclient import TestClient

from campus_nav.api.main import create_app
from campus_nav.infrastructure.dashboard.mock_dashboard_repository import MockDashboardRepository
from campus_nav.interfaces.dashboard_interface import BusSchedule, CityBusSchedule, MessMenu

pytestmark = pytest.mark.integration

FIXTURE_GRAPH = "tests/fixtures/sample_campus_graph.json"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")

    def factory():
        return MockDashboardRepository(
            bus_schedule=BusSchedule(routes={"Gate to Hostel": ["08:00:00", "08:30:00"]}),
            city_bus_schedule=CityBusSchedule(raw={"Route 1": ["09:00"]}),
            mess_menu=MessMenu(days={"Monday": {"Breakfast": ["Idli", "Sambar"]}}),
        )

    app = create_app(graph_path=FIXTURE_GRAPH, dashboard_repository_factory=factory)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def unavailable_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test2.db")

    def factory():
        return MockDashboardRepository(available=False)

    app = create_app(graph_path=FIXTURE_GRAPH, dashboard_repository_factory=factory)
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_status(client):
    response = client.get("/dashboard/status")
    assert response.status_code == 200
    assert response.json()["is_available"] is True


def test_bus_schedule_endpoint(client):
    response = client.get("/dashboard/bus-schedule")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["routes"] == {"Gate to Hostel": ["08:00:00", "08:30:00"]}


def test_city_bus_schedule_endpoint(client):
    response = client.get("/dashboard/city-bus-schedule")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["raw"] == {"Route 1": ["09:00"]}


def test_mess_menu_endpoint(client):
    response = client.get("/dashboard/mess-menu")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["days"] == {"Monday": {"Breakfast": ["Idli", "Sambar"]}}


def test_unavailable_dashboard_returns_200_with_available_false(unavailable_client):
    response = unavailable_client.get("/dashboard/bus-schedule")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["routes"] == {}


def test_unavailable_dashboard_status_reflects_unavailability(unavailable_client):
    response = unavailable_client.get("/dashboard/status")
    assert response.status_code == 200
    assert response.json()["is_available"] is False


def test_default_app_without_injected_factory_uses_mock_backend(tmp_path, monkeypatch):
    """Regression guard: without an explicit dashboard_repository_factory,
    create_app must still boot cleanly using Settings' mock default --
    no network access, no crash."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test3.db")
    app = create_app(graph_path=FIXTURE_GRAPH)
    with TestClient(app) as test_client:
        response = test_client.get("/dashboard/status")
        assert response.status_code == 200
        assert response.json()["backend"] == "mock"
