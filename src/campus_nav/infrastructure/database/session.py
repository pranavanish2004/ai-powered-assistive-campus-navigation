"""
Database engine and session management.

A single SQLAlchemy `engine` is built from `Settings.database_url`, so
moving from SQLite (pilot) to PostgreSQL (multi-user production) is an env
var change, not a code change. `get_db()` is a generator-based dependency
matching FastAPI's DI pattern; `session_scope()` is the context-manager form
for use outside request handlers (background tasks, scripts, tests).
"""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from campus_nav.config.settings import get_settings
from campus_nav.core.exceptions import DatabaseError
from campus_nav.core.logging_config import get_logger
from campus_nav.infrastructure.database.models import Base

logger = get_logger(__name__)


def _ensure_sqlite_directory_exists(database_url: str) -> None:
    """SQLite will not create missing parent directories itself and fails
    with an opaque 'unable to open database file' error instead. Since the
    default database_url is a relative path, this must not assume the
    process's current working directory is the project root (e.g. tests or
    a WSGI server may run from elsewhere)."""
    match = re.match(r"sqlite:///(.+)", database_url)
    if not match:
        return
    db_path = Path(match.group(1))
    if db_path == Path(":memory:"):
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)


def build_engine(database_url: str | None = None) -> Engine:
    """Build a SQLAlchemy engine. Exposed as a function (rather than only a
    module-level singleton) so tests can build isolated engines against
    temporary databases without touching the app's configured database.

    NOTE on lifecycle: `create_engine` opens a connection pool, and for
    SQLite that pool holds an OS-level file handle open on the database
    file until the engine is disposed. The long-lived module-level `engine`
    below is intentionally never disposed during normal app operation (it
    lives for the process lifetime). Callers that build *short-lived*
    engines of their own -- e.g. tests pointing at a temporary directory --
    are responsible for disposing them (see `engine_scope` below), or the
    pooled connection will keep the underlying file open. On POSIX this is
    usually invisible (unlinking an open file succeeds), but on Windows an
    open file handle prevents the file -- and therefore its containing
    temporary directory -- from being deleted, raising PermissionError.
    """
    settings = get_settings()
    url = database_url or settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        _ensure_sqlite_directory_exists(url)
        # Required for SQLite when accessed from multiple threads (FastAPI
        # may service requests on different threads than they were created).
        connect_args = {"check_same_thread": False}
    return create_engine(url, connect_args=connect_args)


@contextmanager
def engine_scope(database_url: str) -> Generator[Engine, None, None]:
    """Build a short-lived engine and guarantee `engine.dispose()` runs on
    exit, even if an exception is raised.

    `engine.dispose()` closes every pooled connection and releases the
    underlying OS file handle(s). This matters specifically for SQLite
    files created inside a `tempfile.TemporaryDirectory()`: without an
    explicit `dispose()`, the pool can keep the file open after the test
    body finishes, and `TemporaryDirectory.__exit__` (or any other cleanup
    that tries to remove the directory) fails on Windows with
    `PermissionError: The process cannot access the file because it is
    being used by another process.` POSIX systems generally don't surface
    this because unlinking an open file is allowed there -- which is why
    the underlying resource leak went unnoticed until tested on Windows.

    Use this instead of a bare `build_engine(...)` call whenever the engine
    is scoped to a single test, request, or short-lived task rather than
    the whole process (the module-level `engine` singleton below is the
    process-lifetime exception to this rule).
    """
    engine_instance = build_engine(database_url)
    try:
        yield engine_instance
    finally:
        engine_instance.dispose()


engine = build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(bind: Engine | None = None) -> None:
    """Create all tables registered on `Base.metadata`. Safe to call
    repeatedly (no-op for existing tables)."""
    try:
        Base.metadata.create_all(bind=bind or engine)
        logger.info("Database schema initialized", extra={"url": str((bind or engine).url)})
    except Exception as exc:
        raise DatabaseError(f"Failed to initialize database schema: {exc}") from exc


def get_db() -> Generator[Session, None, None]:
    """FastAPI-style dependency: yields a session, guarantees closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope(session_factory: sessionmaker | None = None) -> Generator[Session, None, None]:
    """Context-manager form of `get_db`, for use outside FastAPI request
    handlers (background tasks, CLI scripts, tests)."""
    factory = session_factory or SessionLocal
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
