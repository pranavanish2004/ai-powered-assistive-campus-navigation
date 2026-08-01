"""
FastAPI application factory.

Design rationale
-----------------
`create_app()` is a factory (not a bare module-level `app = FastAPI()`) so
tests can construct independent app instances with isolated graph/database
configuration, rather than sharing global state across the test suite.

Startup responsibilities, in order:
1. Configure logging.
2. Initialize the database schema.
3. Load and validate the campus graph once, stored on `app.state.graph`.
4. Create the (empty) in-memory navigation engine registry on
   `app.state.navigation_engines` (see api/dependencies.py for its
   documented scaling limitation).

Domain exceptions (NodeNotFoundError, NoRouteFoundError, etc.) are mostly
caught explicitly in each router for precise HTTP status codes; the handlers
registered here are a safety net so an *unexpected* CampusNavError subclass
still returns a clean JSON error instead of a raw 500 traceback leaking to
the client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from campus_nav.config.settings import get_settings
from campus_nav.core.exceptions import CampusNavError, FatalError
from campus_nav.core.logging_config import configure_logging, get_logger
from campus_nav.infrastructure.database.session import init_db
from campus_nav.infrastructure.graph.json_repository import JSONGraphRepository

logger = get_logger(__name__)

_DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[3] / "data" / "sample_campus_graph.json"


def create_app(graph_path: str | Path | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    `graph_path` is exposed as a parameter (rather than always reading from
    Settings) specifically so tests can point at a small, isolated fixture
    graph instead of the full pilot campus graph.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        init_db()

        repository = JSONGraphRepository(graph_path or _DEFAULT_GRAPH_PATH)
        app.state.graph = repository.load()
        app.state.navigation_engines = {}

        logger.info(
            "Application startup complete",
            extra={
                "nodes": app.state.graph.node_count(),
                "edges": app.state.graph.edge_count(),
            },
        )
        yield
        logger.info("Application shutdown")

    app = FastAPI(
        title="Campus Navigation Assistant API",
        version="0.2.0",
        description="Navigation Core (Phase 1.2) API for the AI Campus "
        "Navigation Assistant pilot at IIT Hyderabad.",
        lifespan=lifespan,
    )

    from campus_nav.api.routers import graph as graph_router
    from campus_nav.api.routers import navigation as navigation_router

    app.include_router(graph_router.router)
    app.include_router(navigation_router.router)

    @app.get("/health", tags=["health"])
    def health_check() -> dict:
        settings = get_settings()
        return {"status": "ok", "app_env": settings.app_env}

    @app.exception_handler(FatalError)
    async def handle_fatal_error(request: Request, exc: FatalError) -> JSONResponse:
        logger.error("Unhandled FatalError", extra={"path": request.url.path}, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(CampusNavError)
    async def handle_campus_nav_error(request: Request, exc: CampusNavError) -> JSONResponse:
        logger.warning(
            "Unhandled CampusNavError", extra={"path": request.url.path, "error": str(exc)}
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
