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

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from campus_nav.config.settings import get_settings
from campus_nav.core.exceptions import CampusNavError, FatalError
from campus_nav.core.logging_config import configure_logging, get_logger
from campus_nav.infrastructure.camera.camera_factory import create_camera
from campus_nav.infrastructure.database.session import init_db
from campus_nav.infrastructure.graph.json_repository import JSONGraphRepository
from campus_nav.vision.camera.camera_manager import CameraManager
from campus_nav.vision.pipeline.vision_factory import create_vision_pipeline
from campus_nav.vision.pipeline.vision_service import VisionService

logger = get_logger(__name__)

_DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[3] / "data" / "sample_campus_graph.json"


def create_app(
    graph_path: str | Path | None = None,
    vision_service_factory: Callable[[], VisionService] | None = None,
) -> FastAPI:
    """Build and return a configured FastAPI application.

    `graph_path` is exposed as a parameter (rather than always reading from
    Settings) specifically so tests can point at a small, isolated fixture
    graph instead of the full pilot campus graph.

    `vision_service_factory`, if provided, is used instead of the default
    production wiring (real camera + real/mock backends per Settings) to
    build `app.state.vision_service`. This is what lets integration tests
    exercise the full /vision/* API surface -- start, stop, detections,
    obstacles -- against a `FakeCamera` and `MockObjectDetector` instead of
    requiring physical camera hardware to be attached to the test runner.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        init_db()

        repository = JSONGraphRepository(graph_path or _DEFAULT_GRAPH_PATH)
        app.state.graph = repository.load()
        app.state.navigation_engines = {}

        # Phase 1.3: constructed at startup but NOT started -- running
        # camera capture and inference continuously is expensive and
        # should only begin on an explicit POST /vision/start call, not
        # merely because the server process booted (see VisionService's
        # docstring for the rationale).
        if vision_service_factory is not None:
            app.state.vision_service = vision_service_factory()
        else:
            settings = get_settings()
            camera_manager = CameraManager(
                camera=create_camera(settings),
                buffer_size=settings.frame_buffer_size,
            )
            app.state.vision_service = VisionService(
                camera_manager=camera_manager,
                pipeline=create_vision_pipeline(settings),
                safe_distance_meters=settings.depth_safe_distance_meters,
                target_fps=settings.vision_processing_fps,
            )

        logger.info(
            "Application startup complete",
            extra={
                "nodes": app.state.graph.node_count(),
                "edges": app.state.graph.edge_count(),
            },
        )
        yield
        app.state.vision_service.stop()
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
    from campus_nav.api.routers import vision as vision_router

    app.include_router(graph_router.router)
    app.include_router(navigation_router.router)
    app.include_router(vision_router.router)

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
