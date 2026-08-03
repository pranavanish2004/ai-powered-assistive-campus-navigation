"""Integration tests for the /vision/* API endpoints, using an injected
fake camera + mock vision backends via create_app's vision_service_factory
parameter (see api/main.py) -- no physical camera hardware required."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from campus_nav.api.main import create_app
from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector, make_detection
from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.interfaces.camera_interface import CameraInterface
from campus_nav.vision.camera.camera_manager import CameraManager
from campus_nav.vision.detection.categories import ObjectCategory
from campus_nav.vision.pipeline.vision_pipeline import VisionPipeline
from campus_nav.vision.pipeline.vision_service import VisionService
from campus_nav.vision.tracking.tracker import IOUTracker

pytestmark = pytest.mark.integration

FIXTURE_GRAPH = "tests/fixtures/sample_campus_graph.json"


class FakeCamera(CameraInterface):
    def __init__(self) -> None:
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8) if self._started else None

    def is_available(self) -> bool:
        return self._started


def _vision_service_factory(detections=None):
    def factory():
        detector = MockObjectDetector(detections or [])
        pipeline = VisionPipeline(
            detector=detector,
            tracker=IOUTracker(),
            depth_estimator=HeuristicDepthEstimator(480),
            ocr=MockOCRService(),
            landmark_detector=NullLandmarkDetector(),
        )
        camera_manager = CameraManager(FakeCamera(), buffer_size=5, target_fps=30.0)
        return VisionService(
            camera_manager=camera_manager,
            pipeline=pipeline,
            safe_distance_meters=1.5,
            target_fps=20.0,
        )

    return factory


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    detection = make_detection(ObjectCategory.STAIRS, "stairs", 0.9, (280, 200, 360, 470))
    app = create_app(
        graph_path=FIXTURE_GRAPH, vision_service_factory=_vision_service_factory([detection])
    )
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_processing(client, timeout_seconds=2.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = client.get("/vision/status").json()
        if status["frames_processed"] > 0:
            return
        time.sleep(0.02)


def test_vision_status_before_start(client):
    response = client.get("/vision/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is False
    assert data["detector_backend"] == "mock"


def test_start_and_stop_vision(client):
    start_response = client.post("/vision/start")
    assert start_response.status_code == 200

    status_response = client.get("/vision/status")
    assert status_response.json()["is_running"] is True

    stop_response = client.post("/vision/stop")
    assert stop_response.status_code == 200

    status_after_stop = client.get("/vision/status")
    assert status_after_stop.json()["is_running"] is False


def test_camera_status_endpoint(client):
    client.post("/vision/start")
    _wait_for_processing(client)
    try:
        response = client.get("/vision/camera/status")
        assert response.status_code == 200
        data = response.json()
        assert data["is_running"] is True
        assert data["frames_captured"] > 0
    finally:
        client.post("/vision/stop")


def test_detections_endpoint_reflects_configured_detection(client):
    client.post("/vision/start")
    _wait_for_processing(client)
    try:
        response = client.get("/vision/detections")
        assert response.status_code == 200
        data = response.json()
        assert len(data["objects"]) == 1
        assert data["objects"][0]["category"] == "stairs"
    finally:
        client.post("/vision/stop")


def test_obstacles_endpoint_reflects_always_announce_policy(client):
    client.post("/vision/start")
    _wait_for_processing(client)
    try:
        response = client.get("/vision/obstacles")
        assert response.status_code == 200
        data = response.json()
        assert len(data["obstacles"]) == 1
        assert data["obstacles"][0]["announce_text"] == "Stairs ahead."
    finally:
        client.post("/vision/stop")


def test_diagnostics_endpoint(client):
    client.post("/vision/start")
    _wait_for_processing(client)
    try:
        response = client.get("/vision/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert data["vision"]["is_running"] is True
        assert data["camera"]["is_running"] is True
        assert data["last_frame_dimensions"] == [640, 480]
    finally:
        client.post("/vision/stop")


def test_detections_endpoint_before_start_returns_empty(client):
    response = client.get("/vision/detections")
    assert response.status_code == 200
    data = response.json()
    assert data["objects"] == []
    assert data["timestamp"] is None


def test_obstacles_endpoint_before_start_returns_empty(client):
    response = client.get("/vision/obstacles")
    assert response.status_code == 200
    assert response.json()["obstacles"] == []
