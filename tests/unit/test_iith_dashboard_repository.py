"""Tests for IITHDashboardRepository, mocking httpx.get since no real
network access to a third-party, no-SLA service should ever be part of
the test suite -- consistent with how cv2.VideoCapture and Whisper/YOLO
model loading are mocked elsewhere in this codebase."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from campus_nav.infrastructure.dashboard.iith_dashboard_repository import IITHDashboardRepository

pytestmark = pytest.mark.unit


def _mock_response(json_data, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_get_bus_schedule_success(mock_get):
    mock_get.return_value = _mock_response(
        {"bus": {"Gate to Hostel": ["08:00:00"]}, "EV": {"Route A": ["09:00:00"]}}
    )
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    schedule = repo.get_bus_schedule()

    assert schedule is not None
    assert schedule.routes == {"Gate to Hostel": ["08:00:00"]}
    assert schedule.ev_routes == {"Route A": ["09:00:00"]}
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert called_url == "https://api.iith.dev/transport/"


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_get_bus_schedule_network_failure_returns_none(mock_get):
    mock_get.side_effect = httpx.ConnectTimeout("simulated timeout")
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    assert repo.get_bus_schedule() is None
    assert repo.is_available() is False


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_get_bus_schedule_http_error_status_returns_none(mock_get):
    mock_get.return_value = _mock_response({}, status_code=500)
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    assert repo.get_bus_schedule() is None


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_get_city_bus_schedule_success(mock_get):
    mock_get.return_value = _mock_response({"some": "shape"})
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    schedule = repo.get_city_bus_schedule()

    assert schedule is not None
    assert schedule.raw == {"some": "shape"}
    called_url = mock_get.call_args[0][0]
    assert called_url == "https://api.iith.dev/transport/cityBus"


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_get_mess_menu_success(mock_get):
    mock_get.return_value = _mock_response({"Monday": {"Breakfast": ["Idli"]}})
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    menu = repo.get_mess_menu()

    assert menu is not None
    assert menu.days == {"Monday": {"Breakfast": ["Idli"]}}
    called_url = mock_get.call_args[0][0]
    assert called_url == "https://api.iith.dev/mess_menu/"


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_get_mess_menu_malformed_shape_returns_none(mock_get):
    mock_get.return_value = _mock_response(["not", "a", "dict"])
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    assert repo.get_mess_menu() is None


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_is_available_reflects_most_recent_call(mock_get):
    repo = IITHDashboardRepository(base_url="https://api.iith.dev")

    mock_get.side_effect = httpx.ConnectTimeout("down")
    repo.get_bus_schedule()
    assert repo.is_available() is False

    mock_get.side_effect = None
    mock_get.return_value = _mock_response({"bus": {}, "EV": {}})
    repo.get_bus_schedule()
    assert repo.is_available() is True


def test_base_url_trailing_slash_is_normalized():
    repo = IITHDashboardRepository(base_url="https://api.iith.dev/")
    assert repo._base_url == "https://api.iith.dev"


@patch("campus_nav.infrastructure.dashboard.iith_dashboard_repository.httpx.get")
def test_timeout_is_passed_through_to_httpx(mock_get):
    mock_get.return_value = _mock_response({"bus": {}, "EV": {}})
    repo = IITHDashboardRepository(base_url="https://api.iith.dev", timeout_seconds=2.5)

    repo.get_bus_schedule()

    assert mock_get.call_args.kwargs["timeout"] == 2.5
