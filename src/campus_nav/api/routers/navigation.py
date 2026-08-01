"""
Navigation endpoints.

Each POST /navigation/sessions call creates one NavigationEngine instance
(held in the in-memory registry, see api/dependencies.py) and a matching
NavigationSessionRecord/RouteRecord pair in the database (for durable
history/analytics -- the engine itself is the source of truth for live
navigation state; the database is a record of what happened, not what's
currently being queried).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from campus_nav.api.dependencies import get_engine_registry, get_graph, get_navigation_engine
from campus_nav.api.schemas.navigation import (
    MultiRouteRequest,
    MultiRouteResponse,
    NavigationStatusResponse,
    PositionUpdateRequest,
    RouteResponse,
    SimpleMessageResponse,
    StartNavigationRequest,
    StartNavigationResponse,
)
from campus_nav.core.exceptions import (
    InvalidStateTransitionError,
    NavigationError,
    NodeNotFoundError,
    NoRouteFoundError,
)
from campus_nav.core.logging_config import get_logger
from campus_nav.infrastructure.database.models import (
    NavigationEventRecord,
    NavigationSessionRecord,
    RouteRecord,
    UserSession,
)
from campus_nav.infrastructure.database.session import get_db
from campus_nav.infrastructure.speech.null_tts import NullTTS
from campus_nav.navigation.engine.navigation_engine import NavigationEngine
from campus_nav.navigation.engine.voice_guidance import VoiceGuidance
from campus_nav.navigation.graph.models import CampusGraph
from campus_nav.navigation.routing.multi_route import k_shortest_routes

router = APIRouter(prefix="/navigation", tags=["navigation"])
logger = get_logger(__name__)


@router.post("/sessions", response_model=StartNavigationResponse, status_code=201)
def start_navigation(
    request: StartNavigationRequest,
    graph: CampusGraph = Depends(get_graph),
    registry: dict = Depends(get_engine_registry),
    db: Session = Depends(get_db),
) -> StartNavigationResponse:
    user_session_id = request.user_session_id
    if user_session_id is None:
        user_session = UserSession()
        db.add(user_session)
        db.flush()
        user_session_id = user_session.id
    elif db.get(UserSession, user_session_id) is None:
        raise HTTPException(status_code=404, detail=f"No such user_session_id: {user_session_id!r}")

    engine = NavigationEngine(graph=graph, voice_guidance=VoiceGuidance(NullTTS()))

    try:
        route = engine.start_navigation(
            request.origin_node_id, request.destination_node_id, request.accessibility_mode
        )
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NoRouteFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    navigation_session_id = str(uuid.uuid4())
    registry[navigation_session_id] = engine

    nav_record = NavigationSessionRecord(
        id=navigation_session_id,
        user_session_id=user_session_id,
        origin_node_id=request.origin_node_id,
        destination_node_id=request.destination_node_id,
        status=engine.state.value,
    )
    db.add(nav_record)
    db.add(
        RouteRecord(
            navigation_session_id=navigation_session_id,
            node_ids_json=json.dumps(route.node_ids),
            algorithm=route.algorithm,
            total_distance_meters=route.total_distance_meters,
            accessible_only=request.accessibility_mode.value == "strict",
        )
    )
    db.commit()

    logger.info(
        "Navigation session started",
        extra={"navigation_session_id": navigation_session_id, "user_session_id": user_session_id},
    )

    return StartNavigationResponse(
        navigation_session_id=navigation_session_id,
        route=RouteResponse.from_domain(route),
        first_instruction=engine.last_instruction.text if engine.last_instruction else None,
    )


@router.post("/routes/alternatives", response_model=MultiRouteResponse)
def get_route_alternatives(
    request: MultiRouteRequest, graph: CampusGraph = Depends(get_graph)
) -> MultiRouteResponse:
    try:
        routes = k_shortest_routes(
            graph,
            request.origin_node_id,
            request.destination_node_id,
            k=request.max_routes,
            accessibility_mode=request.accessibility_mode,
        )
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return MultiRouteResponse(routes=[RouteResponse.from_domain(r) for r in routes])


@router.post("/sessions/{navigation_session_id}/position", response_model=NavigationStatusResponse)
def update_position(
    navigation_session_id: str,
    request: PositionUpdateRequest,
    engine: NavigationEngine = Depends(get_navigation_engine),
    db: Session = Depends(get_db),
) -> NavigationStatusResponse:
    try:
        update = engine.update_position(
            request.latitude, request.longitude, request.accuracy_meters
        )
    except NavigationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    nav_record = db.get(NavigationSessionRecord, navigation_session_id)
    if nav_record is not None:
        nav_record.status = update.state.value
        if update.safety_alert:
            db.add(
                NavigationEventRecord(
                    navigation_session_id=navigation_session_id,
                    event_type="safety_alert",
                    message=update.safety_alert,
                    node_id=update.current_node_id,
                )
            )
        db.commit()

    return NavigationStatusResponse(
        state=update.state,
        current_node_id=update.current_node_id,
        next_instruction=update.next_instruction.text if update.next_instruction else None,
        safety_alert=update.safety_alert,
        distance_remaining_meters=update.distance_remaining_meters,
    )


@router.post("/sessions/{navigation_session_id}/pause", response_model=SimpleMessageResponse)
def pause_navigation(
    navigation_session_id: str,
    engine: NavigationEngine = Depends(get_navigation_engine),
) -> SimpleMessageResponse:
    try:
        engine.pause()
    except InvalidStateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SimpleMessageResponse(message="Navigation paused.")


@router.post("/sessions/{navigation_session_id}/resume", response_model=SimpleMessageResponse)
def resume_navigation(
    navigation_session_id: str,
    engine: NavigationEngine = Depends(get_navigation_engine),
) -> SimpleMessageResponse:
    try:
        engine.resume()
    except InvalidStateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SimpleMessageResponse(message="Navigation resumed.")


@router.post("/sessions/{navigation_session_id}/stop", response_model=SimpleMessageResponse)
def stop_navigation(
    navigation_session_id: str,
    engine: NavigationEngine = Depends(get_navigation_engine),
    registry: dict = Depends(get_engine_registry),
    db: Session = Depends(get_db),
) -> SimpleMessageResponse:
    try:
        engine.stop()
    except InvalidStateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    nav_record = db.get(NavigationSessionRecord, navigation_session_id)
    if nav_record is not None:
        from datetime import datetime

        nav_record.status = "idle"
        nav_record.ended_at = datetime.now(UTC)
        db.commit()

    registry.pop(navigation_session_id, None)
    return SimpleMessageResponse(message="Navigation stopped.")


@router.get("/sessions/{navigation_session_id}/status", response_model=NavigationStatusResponse)
def get_status(
    navigation_session_id: str,
    engine: NavigationEngine = Depends(get_navigation_engine),
) -> NavigationStatusResponse:
    route = engine.active_route
    return NavigationStatusResponse(
        state=engine.state,
        current_node_id=engine.last_known_node_id,
        next_instruction=engine.last_instruction.text if engine.last_instruction else None,
        safety_alert=None,
        distance_remaining_meters=route.total_distance_meters if route else None,
    )
