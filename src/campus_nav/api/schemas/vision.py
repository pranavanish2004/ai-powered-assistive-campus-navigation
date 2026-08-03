"""Pydantic request/response schemas for the /vision API endpoints.

Deliberately separate from the domain dataclasses in `vision/` and
`interfaces/`, matching the pattern already established by
`api/schemas/graph.py` and `api/schemas/navigation.py` -- the API's wire
format is allowed to evolve independently of the internal vision pipeline
types.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from campus_nav.interfaces.depth_interface import DepthEstimate
from campus_nav.interfaces.obstacle_interface import ObstacleEvent, ObstaclePosition
from campus_nav.interfaces.tracking_interface import TrackedObject
from campus_nav.vision.camera.camera_manager import CameraManagerDiagnostics
from campus_nav.vision.detection.categories import ObjectCategory
from campus_nav.vision.pipeline.vision_service import VisionServiceDiagnostics


class BoundingBoxResponse(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class CameraStatusResponse(BaseModel):
    is_running: bool
    frames_captured: int
    frames_dropped: int
    last_capture_error: str | None = None
    actual_fps: float

    @classmethod
    def from_domain(cls, diagnostics: CameraManagerDiagnostics) -> CameraStatusResponse:
        return cls(
            is_running=diagnostics.is_running,
            frames_captured=diagnostics.frames_captured,
            frames_dropped=diagnostics.frames_dropped,
            last_capture_error=diagnostics.last_capture_error,
            actual_fps=diagnostics.actual_fps,
        )


class VisionStatusResponse(BaseModel):
    is_running: bool
    frames_processed: int
    actual_processing_fps: float
    last_process_error: str | None = None
    last_result_timestamp: str | None = None
    detector_backend: str
    ocr_backend: str
    depth_backend: str

    @classmethod
    def from_domain(
        cls,
        diagnostics: VisionServiceDiagnostics,
        detector_backend: str,
        ocr_backend: str,
        depth_backend: str,
    ) -> VisionStatusResponse:
        return cls(
            is_running=diagnostics.is_running,
            frames_processed=diagnostics.frames_processed,
            actual_processing_fps=diagnostics.actual_processing_fps,
            last_process_error=diagnostics.last_process_error,
            last_result_timestamp=diagnostics.last_result_timestamp,
            detector_backend=detector_backend,
            ocr_backend=ocr_backend,
            depth_backend=depth_backend,
        )


class TrackedObjectResponse(BaseModel):
    track_id: int
    category: ObjectCategory
    label: str
    confidence: float
    bbox: BoundingBoxResponse
    frames_tracked: int
    frames_missed: int
    distance_meters: float | None = None
    depth_confidence: float | None = None

    @classmethod
    def from_domain(
        cls, tracked: TrackedObject, depth: DepthEstimate | None
    ) -> TrackedObjectResponse:
        return cls(
            track_id=tracked.track_id,
            category=tracked.category,
            label=tracked.label,
            confidence=tracked.confidence,
            bbox=BoundingBoxResponse(
                x_min=tracked.bbox.x_min,
                y_min=tracked.bbox.y_min,
                x_max=tracked.bbox.x_max,
                y_max=tracked.bbox.y_max,
            ),
            frames_tracked=tracked.frames_tracked,
            frames_missed=tracked.frames_missed,
            distance_meters=depth.distance_meters if depth else None,
            depth_confidence=depth.confidence if depth else None,
        )


class OCRResultResponse(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBoxResponse


class LandmarkMatchResponse(BaseModel):
    node_id: str
    confidence: float
    bbox: BoundingBoxResponse


class DetectionsResponse(BaseModel):
    timestamp: str | None = None
    objects: list[TrackedObjectResponse] = Field(default_factory=list)
    ocr_results: list[OCRResultResponse] = Field(default_factory=list)
    landmark_matches: list[LandmarkMatchResponse] = Field(default_factory=list)
    processing_time_seconds: float | None = None


class ObstacleResponse(BaseModel):
    track_id: int
    category: ObjectCategory
    label: str
    position: ObstaclePosition
    distance_meters: float | None = None
    confidence: float
    announce_text: str | None = None

    @classmethod
    def from_domain(cls, event: ObstacleEvent) -> ObstacleResponse:
        return cls(
            track_id=event.track_id,
            category=event.category,
            label=event.label,
            position=event.position,
            distance_meters=event.distance_meters,
            confidence=event.confidence,
            announce_text=event.announce_text,
        )


class ObstaclesResponse(BaseModel):
    obstacles: list[ObstacleResponse] = Field(default_factory=list)


class VisionDiagnosticsResponse(BaseModel):
    vision: VisionStatusResponse
    camera: CameraStatusResponse
    last_frame_dimensions: tuple[int, int] | None = None
    last_processing_time_seconds: float | None = None
