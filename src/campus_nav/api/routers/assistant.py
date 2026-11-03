"""
Assistant endpoints (Phase 1.4d/1.4e) -- "Ask AI" conversational interface.

This router is where context actually gets gathered from the other
subsystems (navigation session registry, vision service, dashboard
repository) and assembled into an `AssistantContext` before being handed
to `DecisionEngine.answer()`. `DecisionEngine` itself stays free of any
dependency on `NavigationEngine`, `VisionService`, or the FastAPI request
object -- this router is the seam where those live, already-running
objects get read from and converted into plain data snapshots.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from campus_nav.api.dependencies import (
    get_dashboard_repository,
    get_decision_engine,
    get_engine_registry,
    get_vision_service,
)
from campus_nav.api.schemas.assistant import AskRequest, AskResponse, AssistantStatusResponse
from campus_nav.assistant.context import (
    AssistantContext,
    DashboardContextSnapshot,
    NavigationContextSnapshot,
    VisionContextSnapshot,
)
from campus_nav.assistant.decision_engine import DecisionEngine
from campus_nav.assistant.prompt_builder import is_dashboard_relevant
from campus_nav.config.settings import get_settings
from campus_nav.core.exceptions import DashboardError
from campus_nav.interfaces.dashboard_interface import DashboardRepositoryInterface
from campus_nav.navigation.engine.navigation_engine import NavigationEngine
from campus_nav.vision.pipeline.vision_service import VisionService

router = APIRouter(prefix="/assistant", tags=["assistant"])


def _build_navigation_context(
    navigation_session_id: str | None, registry: dict[str, NavigationEngine]
) -> NavigationContextSnapshot | None:
    if navigation_session_id is None:
        return None
    engine = registry.get(navigation_session_id)
    if engine is None:
        return None

    route = engine.active_route
    return NavigationContextSnapshot(
        state=engine.state.value,
        origin_node_id=route.node_ids[0] if route and route.node_ids else None,
        destination_node_id=route.node_ids[-1] if route and route.node_ids else None,
        current_node_id=engine.last_known_node_id,
        distance_remaining_meters=route.total_distance_meters if route else None,
        next_instruction_text=engine.last_instruction.text if engine.last_instruction else None,
    )


def _build_vision_context(
    include_vision_context: bool, vision_service: VisionService
) -> VisionContextSnapshot | None:
    if not include_vision_context or not vision_service.is_running():
        return None

    obstacles = vision_service.get_latest_obstacles()
    result = vision_service.get_latest_result()
    ocr_texts = [r.text for r in result.ocr_results] if result else []
    return VisionContextSnapshot(obstacles=obstacles, ocr_texts=ocr_texts)


def _build_dashboard_context(
    query: str, dashboard_repository: DashboardRepositoryInterface
) -> DashboardContextSnapshot | None:
    if not is_dashboard_relevant(query):
        return None

    try:
        bus_schedule = dashboard_repository.get_bus_schedule()
        mess_menu = dashboard_repository.get_mess_menu()
    except DashboardError:
        return DashboardContextSnapshot()

    return DashboardContextSnapshot(bus_schedule=bus_schedule, mess_menu=mess_menu)


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    decision_engine: DecisionEngine = Depends(get_decision_engine),
    navigation_registry: dict = Depends(get_engine_registry),
    vision_service: VisionService = Depends(get_vision_service),
    dashboard_repository: DashboardRepositoryInterface = Depends(get_dashboard_repository),
) -> AskResponse:
    context = AssistantContext(
        navigation=_build_navigation_context(request.navigation_session_id, navigation_registry),
        vision=_build_vision_context(request.include_vision_context, vision_service),
        dashboard=_build_dashboard_context(request.query, dashboard_repository),
    )

    response = decision_engine.answer(request.query, context)
    return AskResponse.from_domain(response)


@router.get("/status", response_model=AssistantStatusResponse)
def assistant_status(
    decision_engine: DecisionEngine = Depends(get_decision_engine),
) -> AssistantStatusResponse:
    settings = get_settings()
    return AssistantStatusResponse(
        mllm_provider=settings.mllm_provider,
        rag_backend=settings.rag_backend,
        knowledge_document_count=decision_engine.knowledge_document_count,
    )
