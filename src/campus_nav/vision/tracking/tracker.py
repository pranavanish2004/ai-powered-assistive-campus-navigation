"""
IOU-based object tracker.

Design rationale
-----------------
A pure-Python, dependency-free tracker (no external tracking library) using
greedy IoU (intersection-over-union) matching between this frame's
detections and existing tracks -- the same fundamental approach as
classical trackers like SORT, simplified to what this project actually
needs: persistent ids, brief-occlusion tolerance, and smoothed bounding
boxes, without a Kalman filter's added complexity or dependencies.

Matching algorithm, per frame:
1. Compute IoU between every existing track's last-known bbox and every new
   detection's bbox.
2. Greedily match the highest-IoU pairs first (above `iou_threshold`),
   removing matched tracks/detections from further consideration.
3. Matched tracks: update bbox (smoothed), reset frames_missed to 0,
   increment frames_tracked.
4. Unmatched existing tracks: increment frames_missed; drop the track
   entirely once frames_missed exceeds `max_missed_frames` (this is the
   occlusion-tolerance window -- a track survives a short gap, e.g.
   someone briefly walking between the camera and a tracked bench).
5. Unmatched detections: become new tracks with a fresh, never-reused id.
"""

from __future__ import annotations

from dataclasses import dataclass

from campus_nav.core.exceptions import TrackingError
from campus_nav.interfaces.detection_interface import BoundingBox, Detection
from campus_nav.interfaces.tracking_interface import ObjectTrackerInterface, TrackedObject

# Exponential smoothing factor for bounding-box coordinates: how much
# weight the newest detection gets vs. the track's previous smoothed
# position. Chosen to noticeably reduce single-frame jitter while still
# tracking genuine movement responsively within a few frames.
_SMOOTHING_ALPHA = 0.6


@dataclass
class _Track:
    track_id: int
    category: object
    label: str
    confidence: float
    bbox: BoundingBox
    frames_tracked: int
    frames_missed: int

    def to_tracked_object(self) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            category=self.category,  # type: ignore[arg-type]
            label=self.label,
            confidence=self.confidence,
            bbox=self.bbox,
            frames_tracked=self.frames_tracked,
            frames_missed=self.frames_missed,
        )


def _smooth_bbox(previous: BoundingBox, new: BoundingBox, alpha: float) -> BoundingBox:
    def blend(prev_value: int, new_value: int) -> int:
        return round(alpha * new_value + (1 - alpha) * prev_value)

    return BoundingBox(
        x_min=blend(previous.x_min, new.x_min),
        y_min=blend(previous.y_min, new.y_min),
        x_max=blend(previous.x_max, new.x_max),
        y_max=blend(previous.y_max, new.y_max),
    )


class IOUTracker(ObjectTrackerInterface):
    """Greedy IoU-matching multi-object tracker with occlusion tolerance
    and bounding-box smoothing."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 10) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        try:
            return self._update(detections)
        except Exception as exc:  # noqa: BLE001 - convert any internal failure to a domain error
            raise TrackingError(f"IOUTracker.update failed: {exc}") from exc

    def _update(self, detections: list[Detection]) -> list[TrackedObject]:
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))

        # Build all (track_id, detection_index, iou) candidate pairs above
        # threshold, then greedily accept the strongest matches first.
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for i, detection in enumerate(detections):
                iou = track.bbox.iou(detection.bbox)
                if iou >= self._iou_threshold:
                    candidates.append((iou, track_id, i))

        candidates.sort(key=lambda c: c[0], reverse=True)

        for _iou, track_id, detection_index in candidates:
            if (
                track_id not in unmatched_track_ids
                or detection_index not in unmatched_detection_indices
            ):
                continue  # one side already matched this round

            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.bbox = _smooth_bbox(track.bbox, detection.bbox, _SMOOTHING_ALPHA)
            track.category = detection.category
            track.label = detection.label
            track.confidence = detection.confidence
            track.frames_tracked += 1
            track.frames_missed = 0

            unmatched_track_ids.discard(track_id)
            unmatched_detection_indices.discard(detection_index)

        # Unmatched existing tracks: age them, dropping any that have
        # exceeded the occlusion-tolerance window.
        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.frames_missed += 1
            if track.frames_missed > self._max_missed_frames:
                del self._tracks[track_id]

        # Unmatched detections: start new tracks.
        for detection_index in unmatched_detection_indices:
            detection = detections[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                category=detection.category,
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
                frames_tracked=1,
                frames_missed=0,
            )

        return [t.to_tracked_object() for t in self._tracks.values()]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
