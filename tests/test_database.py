"""
Tests for the database infrastructure layer.

Uses an isolated in-memory SQLite engine (not the app's configured
database_url) so tests never touch a developer's real data/campus_nav.db.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from campus_nav.infrastructure.database.models import Base, SystemEvent, UserSession


def _make_test_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def test_user_session_round_trip():
    session_factory = _make_test_session_factory()
    db = session_factory()
    try:
        session = UserSession()
        db.add(session)
        db.commit()

        fetched = db.get(UserSession, session.id)
        assert fetched is not None
        assert fetched.started_at is not None
        assert fetched.ended_at is None
    finally:
        db.close()


def test_system_event_linked_to_session():
    session_factory = _make_test_session_factory()
    db = session_factory()
    try:
        session = UserSession()
        db.add(session)
        db.commit()

        event = SystemEvent(
            session_id=session.id,
            event_type="session_started",
            description="Navigation session started for testing.",
        )
        db.add(event)
        db.commit()

        fetched = db.get(SystemEvent, event.id)
        assert fetched is not None
        assert fetched.session_id == session.id
        assert fetched.event_type == "session_started"
    finally:
        db.close()
