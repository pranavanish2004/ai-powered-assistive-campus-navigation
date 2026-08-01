"""Tests for infrastructure.database.session: engine building, schema init,
and the session_scope/get_db dependency helpers."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from campus_nav.core.exceptions import DatabaseError
from campus_nav.infrastructure.database.models import UserSession
from campus_nav.infrastructure.database.session import engine_scope, init_db, session_scope

pytestmark = pytest.mark.unit


def test_build_engine_creates_missing_sqlite_parent_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "nested" / "does" / "not" / "exist" / "test.db"
        with engine_scope(f"sqlite:///{db_path}") as engine:
            init_db(bind=engine)
            assert db_path.exists()
        # engine_scope disposes the engine on exit, releasing SQLite's file
        # handle before the TemporaryDirectory below tries to remove it --
        # required on Windows, where an open handle blocks deletion.


def test_init_db_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        with engine_scope(f"sqlite:///{db_path}") as engine:
            init_db(bind=engine)
            init_db(bind=engine)  # should not raise


def test_session_scope_commits_on_success():
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        engine_scope(f"sqlite:///{tmp_dir}/test.db") as engine,
    ):
        init_db(bind=engine)
        factory = sessionmaker(bind=engine)

        with session_scope(factory) as db:
            db.add(UserSession(id="test-1"))

        with session_scope(factory) as db:
            assert db.get(UserSession, "test-1") is not None


def test_session_scope_rolls_back_on_exception():
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        engine_scope(f"sqlite:///{tmp_dir}/test.db") as engine,
    ):
        init_db(bind=engine)
        factory = sessionmaker(bind=engine)

        with pytest.raises(ValueError), session_scope(factory) as db:
            db.add(UserSession(id="test-2"))
            raise ValueError("simulated failure")

        with session_scope(factory) as db:
            assert db.get(UserSession, "test-2") is None


def test_init_db_wraps_failures_as_database_error(monkeypatch):
    import campus_nav.infrastructure.database.session as session_module

    class ExplodingMetadata:
        def create_all(self, bind):
            raise RuntimeError("disk on fire")

    monkeypatch.setattr(session_module.Base, "metadata", ExplodingMetadata())
    with pytest.raises(DatabaseError):
        init_db()
