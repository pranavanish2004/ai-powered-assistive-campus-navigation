"""Tests for localization confidence estimation."""

import pytest

from campus_nav.navigation.localization.confidence import (
    combined_localization_confidence,
    confidence_from_gps_accuracy,
    confidence_from_snap_distance,
)

pytestmark = pytest.mark.unit


def test_confidence_from_snap_distance_zero_distance_is_full_confidence():
    assert confidence_from_snap_distance(0.0, 30.0) == pytest.approx(1.0)


def test_confidence_from_snap_distance_at_max_is_zero():
    assert confidence_from_snap_distance(30.0, 30.0) == pytest.approx(0.0)


def test_confidence_from_snap_distance_beyond_max_clamped_to_zero():
    assert confidence_from_snap_distance(100.0, 30.0) == 0.0


def test_confidence_from_snap_distance_zero_max_distance_is_zero():
    assert confidence_from_snap_distance(5.0, 0.0) == 0.0


def test_confidence_from_gps_accuracy_precise_fix_is_high_confidence():
    assert confidence_from_gps_accuracy(1.0, reference_accuracy_meters=10.0) == pytest.approx(0.9)


def test_confidence_from_gps_accuracy_imprecise_fix_clamped_to_zero():
    assert confidence_from_gps_accuracy(50.0, reference_accuracy_meters=10.0) == 0.0


def test_combined_confidence_takes_minimum_not_average():
    # Perfect snap (confidence 1.0) but terrible GPS accuracy (confidence 0.0)
    # -- combined must reflect the weaker signal, not average to 0.5.
    combined = combined_localization_confidence(
        snap_distance_meters=0.0,
        max_snap_distance_meters=30.0,
        gps_accuracy_meters=100.0,
        reference_gps_accuracy_meters=10.0,
    )
    assert combined == pytest.approx(0.0)


def test_combined_confidence_both_good_is_high():
    combined = combined_localization_confidence(
        snap_distance_meters=1.0,
        max_snap_distance_meters=30.0,
        gps_accuracy_meters=1.0,
        reference_gps_accuracy_meters=10.0,
    )
    assert combined > 0.8
