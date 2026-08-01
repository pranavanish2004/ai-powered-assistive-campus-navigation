"""Tests for localize_fix() and GraphSnappingLocalizer."""

import pytest

from campus_nav.infrastructure.gps.mock_gps import MockGPSProvider
from campus_nav.navigation.localization.localizer import GraphSnappingLocalizer, localize_fix

pytestmark = pytest.mark.unit


def test_localize_fix_returns_result_when_within_range(simple_graph):
    result = localize_fix(
        simple_graph, 17.590, 78.120, accuracy_meters=5.0, max_snap_distance_meters=50.0
    )
    assert result is not None
    assert result.nearest_node.id == "A"
    assert 0.0 <= result.confidence <= 1.0


def test_localize_fix_returns_none_when_too_far(simple_graph):
    result = localize_fix(
        simple_graph, 0.0, 0.0, accuracy_meters=5.0, max_snap_distance_meters=10.0
    )
    assert result is None


def test_graph_snapping_localizer_returns_none_when_gps_unavailable(simple_graph):
    gps = MockGPSProvider(available=False)
    localizer = GraphSnappingLocalizer(gps, max_snap_distance_meters=50.0)
    assert localizer.get_current_position(simple_graph) is None


def test_graph_snapping_localizer_returns_result_when_gps_available(simple_graph):
    gps = MockGPSProvider(available=True)
    gps.set_location(17.590, 78.120)
    localizer = GraphSnappingLocalizer(gps, max_snap_distance_meters=50.0)
    result = localizer.get_current_position(simple_graph)
    assert result is not None
    assert result.nearest_node.id == "A"


def test_graph_snapping_localizer_returns_none_when_fix_too_far(simple_graph):
    gps = MockGPSProvider(available=True)
    gps.set_location(0.0, 0.0)
    localizer = GraphSnappingLocalizer(gps, max_snap_distance_meters=10.0)
    assert localizer.get_current_position(simple_graph) is None
