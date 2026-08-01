"""
FastAPI dependency providers.

Design rationale
-----------------
The campus graph is loaded once at startup (app.state.graph) and shared
read-only across requests -- it is authored data, not per-request state.
The navigation engine registry (app.state.navigation_engines) is a
per-process, in-memory dict keyed by navigation_session_id.

This in-memory registry is a known, explicitly documented scaling
limitation (see docs/architecture.md, Risks): it does not survive a process
restart and does not work across multiple server instances behind a load
balancer. That's an acceptable trade-off for a single-process pilot
deployment, but MUST be replaced with a shared store (Redis, or
database-backed session reconstruction) before any multi-instance
production deployment. Isolating it behind these dependency functions means
that future change touches this one file, not every router.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from campus_nav.navigation.engine.navigation_engine import NavigationEngine
from campus_nav.navigation.graph.models import CampusGraph


def get_graph(request: Request) -> CampusGraph:
    return request.app.state.graph


def get_engine_registry(request: Request) -> dict[str, NavigationEngine]:
    return request.app.state.navigation_engines


def get_navigation_engine(navigation_session_id: str, request: Request) -> NavigationEngine:
    registry = get_engine_registry(request)
    engine = registry.get(navigation_session_id)
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active navigation session with id {navigation_session_id!r}.",
        )
    return engine
