"""
Vision endpoints (Phase 1.3).

Unlike navigation sessions (per-user, created on demand), the vision
pipeline is a single, process-wide resource -- there is one camera on the
pilot device. `POST /vision/start` and `/vision/stop` control it directly
rather than through a session-creation pattern.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from campus_nav.api.dependencies import get_vision_service
from campus_nav.api.schemas.navigation import SimpleMessageResponse
from campus_nav.api.schemas.vision import (
    BoundingBoxResponse,
    CameraStatusResponse,
    DetectionsResponse,
    LandmarkMatchResponse,
    ObstacleResponse,
    ObstaclesResponse,
    OCRResultResponse,
    TrackedObjectResponse,
    VisionDiagnosticsResponse,
    VisionStatusResponse,
)
from campus_nav.config.settings import get_settings
from campus_nav.vision.pipeline.vision_service import VisionService

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/start", response_model=SimpleMessageResponse)
def start_vision(
    vision_service: VisionService = Depends(get_vision_service),
) -> SimpleMessageResponse:
    vision_service.start()
    return SimpleMessageResponse(message="Vision pipeline started.")


@router.post("/stop", response_model=SimpleMessageResponse)
def stop_vision(
    vision_service: VisionService = Depends(get_vision_service),
) -> SimpleMessageResponse:
    vision_service.stop()
    return SimpleMessageResponse(message="Vision pipeline stopped.")


@router.get("/camera/status", response_model=CameraStatusResponse)
def camera_status(
    vision_service: VisionService = Depends(get_vision_service),
) -> CameraStatusResponse:
    diagnostics = vision_service.get_diagnostics()
    return CameraStatusResponse.from_domain(diagnostics.camera)


@router.get("/status", response_model=VisionStatusResponse)
def vision_status(
    vision_service: VisionService = Depends(get_vision_service),
) -> VisionStatusResponse:
    settings = get_settings()
    diagnostics = vision_service.get_diagnostics()
    return VisionStatusResponse.from_domain(
        diagnostics,
        detector_backend=settings.detector_backend,
        ocr_backend=settings.ocr_backend,
        depth_backend=settings.depth_backend,
    )


@router.get("/detections", response_model=DetectionsResponse)
def get_detections(
    vision_service: VisionService = Depends(get_vision_service),
) -> DetectionsResponse:
    result = vision_service.get_latest_result()
    if result is None:
        return DetectionsResponse()

    objects = [
        TrackedObjectResponse.from_domain(tracked, result.depth_estimates.get(tracked.track_id))
        for tracked in result.tracked_objects
    ]
    ocr_results = [
        OCRResultResponse(
            text=r.text,
            confidence=r.confidence,
            bbox=BoundingBoxResponse(
                x_min=r.bbox.x_min, y_min=r.bbox.y_min, x_max=r.bbox.x_max, y_max=r.bbox.y_max
            ),
        )
        for r in result.ocr_results
    ]
    landmark_matches = [
        LandmarkMatchResponse(
            node_id=m.node_id,
            confidence=m.confidence,
            bbox=BoundingBoxResponse(
                x_min=m.bbox.x_min, y_min=m.bbox.y_min, x_max=m.bbox.x_max, y_max=m.bbox.y_max
            ),
        )
        for m in result.landmark_matches
    ]

    return DetectionsResponse(
        timestamp=result.timestamp.isoformat(),
        objects=objects,
        ocr_results=ocr_results,
        landmark_matches=landmark_matches,
        processing_time_seconds=result.processing_time_seconds,
    )


@router.get("/obstacles", response_model=ObstaclesResponse)
def get_obstacles(vision_service: VisionService = Depends(get_vision_service)) -> ObstaclesResponse:
    events = vision_service.get_latest_obstacles()
    return ObstaclesResponse(obstacles=[ObstacleResponse.from_domain(e) for e in events])


@router.get("/diagnostics", response_model=VisionDiagnosticsResponse)
def get_diagnostics(
    vision_service: VisionService = Depends(get_vision_service),
) -> VisionDiagnosticsResponse:
    settings = get_settings()
    diagnostics = vision_service.get_diagnostics()
    result = vision_service.get_latest_result()

    return VisionDiagnosticsResponse(
        vision=VisionStatusResponse.from_domain(
            diagnostics,
            detector_backend=settings.detector_backend,
            ocr_backend=settings.ocr_backend,
            depth_backend=settings.depth_backend,
        ),
        camera=CameraStatusResponse.from_domain(diagnostics.camera),
        last_frame_dimensions=(result.frame_width, result.frame_height) if result else None,
        last_processing_time_seconds=result.processing_time_seconds if result else None,
    )
