"""Integration tests for the /assistant/* API endpoints -- exercising the
full fusion path: navigation session registry, vision service, dashboard
repository, RAG, and MLLM, all wired together as they run for real."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from campus_nav.api.main import create_app
from campus_nav.assistant.decision_engine import DecisionEngine
from campus_nav.infrastructure.dashboard.mock_dashboard_repository import MockDashboardRepository
from campus_nav.infrastructure.depth.heuristic_depth_estimator import HeuristicDepthEstimator
from campus_nav.infrastructure.detection.mock_detector import MockObjectDetector
from campus_nav.infrastructure.landmark.null_landmark_detector import NullLandmarkDetector
from campus_nav.infrastructure.mllm.mock_mllm_provider import MockMLLMProvider
from campus_nav.infrastructure.ocr.mock_ocr import MockOCRService
from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.interfaces.camera_interface import CameraInterface
from campus_nav.interfaces.dashboard_interface import BusSchedule, MessMenu
from campus_nav.interfaces.rag_interface import KnowledgeDocument
from campus_nav.vision.camera.camera_manager import CameraManager
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


def _decision_engine_factory(response_text="Test response."):
    def factory():
        kb = KeywordKnowledgeBase()
        kb.add_document(
            KnowledgeDocument(
                id="lib", title="Central Library", content="The main library on campus."
            )
        )
        mllm = MockMLLMProvider(fixed_response_text=response_text)
        return DecisionEngine(rag_repository=kb, mllm_provider=mllm)

    return factory


def _vision_service_factory(detections=None):
    def factory():
        pipeline = VisionPipeline(
            detector=MockObjectDetector(detections or []),
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


def _dashboard_repository_factory():
    return MockDashboardRepository(
        bus_schedule=BusSchedule(routes={"Gate to Hostel": ["08:00:00"]}),
        mess_menu=MessMenu(days={"Monday": {"Lunch": ["Rice", "Dal"]}}),
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app(
        graph_path=FIXTURE_GRAPH,
        decision_engine_factory=_decision_engine_factory(),
        vision_service_factory=_vision_service_factory(),
        dashboard_repository_factory=_dashboard_repository_factory,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_assistant_status(client):
    response = client.get("/assistant/status")
    assert response.status_code == 200
    data = response.json()
    assert data["mllm_provider"] == "mock"
    assert data["knowledge_document_count"] == 1


def test_ask_basic_query(client):
    response = client.post("/assistant/ask", json={"query": "Where is the library?"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Test response."
    assert data["used_rag"] is True
    assert "lib" in data["source_document_ids"]


def test_ask_with_navigation_context(client):
    start_response = client.post(
        "/navigation/sessions",
        json={"origin_node_id": "main_gate", "destination_node_id": "library"},
    )
    session_id = start_response.json()["navigation_session_id"]

    response = client.post(
        "/assistant/ask",
        json={"query": "How far do I have left?", "navigation_session_id": session_id},
    )
    assert response.status_code == 200
    assert response.json()["used_navigation_context"] is True


def test_ask_with_unknown_navigation_session_id_degrades_gracefully(client):
    response = client.post(
        "/assistant/ask", json={"query": "How far?", "navigation_session_id": "does-not-exist"}
    )
    assert response.status_code == 200
    assert response.json()["used_navigation_context"] is False


def test_ask_with_dashboard_relevant_query(client):
    response = client.post("/assistant/ask", json={"query": "What's for lunch today?"})
    assert response.status_code == 200
    assert response.json()["used_dashboard"] is True


def test_ask_with_non_dashboard_query_does_not_fetch_dashboard(client):
    response = client.post("/assistant/ask", json={"query": "Where is the library?"})
    assert response.status_code == 200
    assert response.json()["used_dashboard"] is False


def test_ask_with_vision_context_when_pipeline_running(client):
    client.post("/vision/start")
    deadline = time.monotonic() + 2.0
    while (
        client.get("/vision/status").json()["frames_processed"] == 0 and time.monotonic() < deadline
    ):
        time.sleep(0.02)

    try:
        response = client.post(
            "/assistant/ask",
            json={"query": "What is in front of me?", "include_vision_context": True},
        )
        assert response.status_code == 200
        assert response.json()["used_vision_context"] is True
    finally:
        client.post("/vision/stop")


def test_ask_with_vision_context_disabled_by_request(client):
    client.post("/vision/start")
    try:
        response = client.post(
            "/assistant/ask",
            json={"query": "What is in front of me?", "include_vision_context": False},
        )
        assert response.status_code == 200
        assert response.json()["used_vision_context"] is False
    finally:
        client.post("/vision/stop")


def test_ask_with_vision_context_when_pipeline_not_running(client):
    response = client.post("/assistant/ask", json={"query": "What is in front of me?"})
    assert response.status_code == 200
    assert response.json()["used_vision_context"] is False


def test_ask_empty_query_rejected(client):
    response = client.post("/assistant/ask", json={"query": ""})
    assert response.status_code == 422


def test_default_app_without_injected_decision_engine_boots_cleanly(tmp_path, monkeypatch):
    """Regression guard: without an explicit decision_engine_factory,
    create_app must still boot using Settings' mock defaults -- no
    network access, no crash."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test2.db")
    app = create_app(graph_path=FIXTURE_GRAPH)
    with TestClient(app) as test_client:
        response = test_client.get("/assistant/status")
        assert response.status_code == 200
        assert response.json()["mllm_provider"] == "mock"
        response = test_client.post("/assistant/ask", json={"query": "Where is the library?"})
        assert response.status_code == 200


def test_default_app_never_returns_old_fixed_placeholder(tmp_path, monkeypatch):
    """Direct regression test for the reported bug: the default (no API
    key, no injected factory) assistant must produce a grounded response,
    never the literal old placeholder string."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test3.db")
    app = create_app(graph_path=FIXTURE_GRAPH)
    with TestClient(app) as test_client:
        for query in [
            "Where is the library?",
            "How do I get to Biomedical Engineering?",
            "What's for lunch today?",
            "Is there an accessible entrance?",
            "asdkjhaskjdh nonsense query",
        ]:
            response = test_client.post("/assistant/ask", json={"query": query})
            assert response.status_code == 200
            assert response.json()["text"] != "This is a mock assistant response."
