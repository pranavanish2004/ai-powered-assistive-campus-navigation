"""Tests for IOUTracker: persistent ids, occlusion tolerance, smoothing."""

import pytest

from campus_nav.core.exceptions import TrackingError
from campus_nav.infrastructure.detection.mock_detector import make_detection
from campus_nav.interfaces.tracking_interface import ObjectTrackerInterface
from campus_nav.vision.detection.categories import ObjectCategory
from campus_nav.vision.tracking.tracker import IOUTracker

pytestmark = pytest.mark.unit


def test_cannot_instantiate_tracker_interface_directly():
    with pytest.raises(TypeError):
        ObjectTrackerInterface()  # type: ignore[abstract]


def test_new_detection_becomes_a_new_track():
    tracker = IOUTracker()
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50))
    tracked = tracker.update([detection])

    assert len(tracked) == 1
    assert tracked[0].track_id == 1
    assert tracked[0].frames_tracked == 1
    assert tracked[0].frames_missed == 0


def test_same_object_across_frames_keeps_same_track_id():
    tracker = IOUTracker(iou_threshold=0.3)
    frame1 = [make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50))]
    frame2 = [make_detection(ObjectCategory.BENCH, "bench", 0.88, (12, 12, 52, 52))]  # slight shift

    tracked1 = tracker.update(frame1)
    tracked2 = tracker.update(frame2)

    assert tracked1[0].track_id == tracked2[0].track_id
    assert tracked2[0].frames_tracked == 2


def test_two_distinct_objects_get_distinct_ids():
    tracker = IOUTracker()
    detections = [
        make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50)),
        make_detection(ObjectCategory.PERSON, "person", 0.85, (300, 300, 350, 400)),
    ]
    tracked = tracker.update(detections)
    ids = {t.track_id for t in tracked}
    assert len(ids) == 2


def test_track_survives_brief_occlusion():
    tracker = IOUTracker(max_missed_frames=3)
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50))

    tracker.update([detection])  # frame 1: appears
    tracked = tracker.update([])  # frame 2: occluded, no detections at all
    assert len(tracked) == 1
    assert tracked[0].frames_missed == 1

    tracked = tracker.update([])  # frame 3: still occluded
    assert len(tracked) == 1
    assert tracked[0].frames_missed == 2


def test_track_dropped_after_exceeding_max_missed_frames():
    tracker = IOUTracker(max_missed_frames=2)
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50))

    tracker.update([detection])  # frame 1
    tracker.update([])  # frame 2: missed 1
    tracker.update([])  # frame 3: missed 2
    tracked = tracker.update([])  # frame 4: missed 3 -- exceeds max, dropped

    assert tracked == []


def test_track_recovers_if_detection_reappears_within_tolerance():
    tracker = IOUTracker(max_missed_frames=3, iou_threshold=0.3)
    detection = make_detection(ObjectCategory.BENCH, "bench", 0.9, (10, 10, 50, 50))

    first = tracker.update([detection])
    tracker.update([])  # occluded briefly
    reappeared = tracker.update([detection])  # same position, reappears

    assert reappeared[0].track_id == first[0].track_id
    assert reappeared[0].frames_missed == 0


def test_bounding_box_is_smoothed_not_snapped():
    tracker = IOUTracker(iou_threshold=0.1)
    frame1 = [make_detection(ObjectCategory.BENCH, "bench", 0.9, (0, 0, 100, 100))]
    frame2 = [
        make_detection(ObjectCategory.BENCH, "bench", 0.9, (40, 40, 140, 140))
    ]  # overlapping shift

    tracked1 = tracker.update(frame1)
    tracked2 = tracker.update(frame2)

    assert tracked1[0].track_id == tracked2[0].track_id  # confirms this is a match, not a new track
    # Smoothed position should be between the two raw detections, not equal
    # to either extreme.
    assert 0 < tracked2[0].bbox.x_min < 40


def test_low_iou_detection_does_not_match_existing_track():
    tracker = IOUTracker(iou_threshold=0.5)
    frame1 = [make_detection(ObjectCategory.BENCH, "bench", 0.9, (0, 0, 50, 50))]
    frame2 = [make_detection(ObjectCategory.BENCH, "bench", 0.9, (500, 500, 550, 550))]

    tracked1 = tracker.update(frame1)
    tracked2 = tracker.update(frame2)

    # The far-away detection should start a new track, and the original
    # track should still exist (missed this frame), so two tracks total.
    assert len(tracked2) == 2
    assert tracked1[0].track_id not in {t.track_id for t in tracked2 if t.frames_missed == 0}


def test_reset_clears_tracks_and_id_counter():
    tracker = IOUTracker()
    tracker.update([make_detection(ObjectCategory.BENCH, "bench", 0.9, (0, 0, 50, 50))])
    tracker.reset()

    tracked = tracker.update([make_detection(ObjectCategory.DOOR, "door", 0.9, (0, 0, 50, 50))])
    assert tracked[0].track_id == 1  # id counter restarted, not continuing from 2


def test_update_wraps_internal_failure_as_tracking_error(monkeypatch):
    tracker = IOUTracker()

    def exploding_update(self, detections):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(IOUTracker, "_update", exploding_update)
    with pytest.raises(TrackingError):
        tracker.update([])
