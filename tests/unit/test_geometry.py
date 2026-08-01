"""Tests for shared geographic math utilities."""

import pytest

from campus_nav.navigation.graph.geometry import (
    bearing_difference_degrees,
    haversine_distance_meters,
    initial_bearing_degrees,
    project_point_onto_segment,
)

pytestmark = pytest.mark.unit


def test_haversine_distance_zero_for_identical_points():
    assert haversine_distance_meters(17.59, 78.12, 17.59, 78.12) == pytest.approx(0.0, abs=1e-6)


def test_haversine_distance_known_value():
    # Roughly 1 degree of latitude ~= 111.32 km.
    distance = haversine_distance_meters(0.0, 0.0, 1.0, 0.0)
    assert distance == pytest.approx(111_195, rel=0.01)


def test_haversine_distance_symmetric():
    d1 = haversine_distance_meters(17.59, 78.12, 17.60, 78.13)
    d2 = haversine_distance_meters(17.60, 78.13, 17.59, 78.12)
    assert d1 == pytest.approx(d2)


def test_initial_bearing_due_north():
    bearing = initial_bearing_degrees(0.0, 0.0, 1.0, 0.0)
    assert bearing == pytest.approx(0.0, abs=0.5)


def test_initial_bearing_due_east():
    bearing = initial_bearing_degrees(0.0, 0.0, 0.0, 1.0)
    assert bearing == pytest.approx(90.0, abs=0.5)


def test_bearing_difference_straight_is_zero():
    assert bearing_difference_degrees(90.0, 90.0) == pytest.approx(0.0)


def test_bearing_difference_right_turn_is_positive():
    assert bearing_difference_degrees(0.0, 90.0) == pytest.approx(90.0)


def test_bearing_difference_left_turn_is_negative():
    assert bearing_difference_degrees(90.0, 0.0) == pytest.approx(-90.0)


def test_bearing_difference_wraparound_normalized():
    # From 350 degrees to 10 degrees is a 20-degree right turn, not -340.
    assert bearing_difference_degrees(350.0, 10.0) == pytest.approx(20.0)


def test_project_point_onto_segment_midpoint():
    # Segment from (0,0) to (0,1) [lat, lon]; point directly "above" the
    # midpoint in longitude should project near the segment's midpoint.
    proj_lat, proj_lon, distance = project_point_onto_segment(0.5, 0.5, 0.0, 0.0, 1.0, 0.0)
    assert proj_lat == pytest.approx(0.5, abs=0.01)
    assert distance > 0


def test_project_point_onto_segment_clamped_before_start():
    # Point "before" A on the line should project onto A itself, not the
    # infinite extension of the line.
    proj_lat, proj_lon, distance = project_point_onto_segment(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert proj_lat == pytest.approx(0.0, abs=0.01)


def test_project_point_onto_segment_clamped_after_end():
    proj_lat, proj_lon, distance = project_point_onto_segment(2.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert proj_lat == pytest.approx(1.0, abs=0.01)


def test_project_point_onto_degenerate_segment():
    # A == B: projection must be point A itself, no division-by-zero crash.
    proj_lat, proj_lon, distance = project_point_onto_segment(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert proj_lat == pytest.approx(0.0, abs=1e-6)
    assert proj_lon == pytest.approx(0.0, abs=1e-6)
    assert distance > 0
