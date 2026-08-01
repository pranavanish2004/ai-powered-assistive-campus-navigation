"""Pydantic request/response schemas for the navigation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from campus_nav.interfaces.navigation_interface import NavState
from campus_nav.navigation.routing.types import AccessibilityMode, Route


class StartNavigationRequest(BaseModel):
    user_session_id: str | None = Field(
        default=None,
        description="Existing UserSession id to attach this navigation session to. "
        "If omitted, a new UserSession is created automatically.",
    )
    origin_node_id: str
    destination_node_id: str
    accessibility_mode: AccessibilityMode = AccessibilityMode.NONE


class RouteResponse(BaseModel):
    node_ids: list[str]
    edge_ids: list[str]
    total_distance_meters: float
    algorithm: str
    accessibility_mode: AccessibilityMode
    uses_inaccessible_edge: bool

    @classmethod
    def from_domain(cls, route: Route) -> RouteResponse:
        return cls(
            node_ids=route.node_ids,
            edge_ids=route.edge_ids,
            total_distance_meters=route.total_distance_meters,
            algorithm=route.algorithm,
            accessibility_mode=route.accessibility_mode,
            uses_inaccessible_edge=route.uses_inaccessible_edge,
        )


class StartNavigationResponse(BaseModel):
    navigation_session_id: str
    route: RouteResponse
    first_instruction: str | None = None


class MultiRouteRequest(BaseModel):
    origin_node_id: str
    destination_node_id: str
    accessibility_mode: AccessibilityMode = AccessibilityMode.NONE
    max_routes: int = Field(default=3, ge=1, le=10)


class MultiRouteResponse(BaseModel):
    routes: list[RouteResponse]


class PositionUpdateRequest(BaseModel):
    latitude: float
    longitude: float
    accuracy_meters: float = Field(default=5.0, gt=0)


class NavigationStatusResponse(BaseModel):
    state: NavState
    current_node_id: str | None = None
    next_instruction: str | None = None
    safety_alert: str | None = None
    distance_remaining_meters: float | None = None


class SimpleMessageResponse(BaseModel):
    message: str
