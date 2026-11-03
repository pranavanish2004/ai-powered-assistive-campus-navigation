"""Tests for the dashboard interface contract and MockDashboardRepository."""

import pytest

from campus_nav.infrastructure.dashboard.mock_dashboard_repository import MockDashboardRepository
from campus_nav.interfaces.dashboard_interface import (
    BusSchedule,
    CityBusSchedule,
    DashboardRepositoryInterface,
    MessMenu,
)

pytestmark = pytest.mark.unit


def test_cannot_instantiate_dashboard_interface_directly():
    with pytest.raises(TypeError):
        DashboardRepositoryInterface()  # type: ignore[abstract]


def test_mock_defaults_to_available_with_no_data_configured():
    repo = MockDashboardRepository()
    assert repo.is_available() is True
    assert repo.get_bus_schedule() is None
    assert repo.get_city_bus_schedule() is None
    assert repo.get_mess_menu() is None


def test_mock_returns_configured_bus_schedule():
    schedule = BusSchedule(routes={"Gate to Hostel": ["08:00:00", "08:30:00"]})
    repo = MockDashboardRepository(bus_schedule=schedule)
    assert repo.get_bus_schedule() == schedule


def test_mock_returns_configured_city_bus_schedule():
    schedule = CityBusSchedule(raw={"route_1": ["09:00"]})
    repo = MockDashboardRepository(city_bus_schedule=schedule)
    assert repo.get_city_bus_schedule() == schedule


def test_mock_returns_configured_mess_menu():
    menu = MessMenu(days={"Monday": {"Breakfast": ["Idli", "Sambar"]}})
    repo = MockDashboardRepository(mess_menu=menu)
    assert repo.get_mess_menu() == menu


def test_mock_unavailable_returns_none_for_everything_even_if_configured():
    schedule = BusSchedule(routes={"X": ["10:00"]})
    menu = MessMenu(days={"Monday": {"Lunch": ["Rice"]}})
    repo = MockDashboardRepository(bus_schedule=schedule, mess_menu=menu, available=False)

    assert repo.is_available() is False
    assert repo.get_bus_schedule() is None
    assert repo.get_mess_menu() is None


def test_set_available_toggles_behavior_dynamically():
    schedule = BusSchedule(routes={"X": ["10:00"]})
    repo = MockDashboardRepository(bus_schedule=schedule)
    assert repo.get_bus_schedule() is not None

    repo.set_available(False)
    assert repo.get_bus_schedule() is None

    repo.set_available(True)
    assert repo.get_bus_schedule() == schedule


def test_bus_schedule_defaults_are_empty_dicts():
    schedule = BusSchedule()
    assert schedule.routes == {}
    assert schedule.ev_routes == {}


def test_mess_menu_default_is_empty_dict():
    assert MessMenu().days == {}
