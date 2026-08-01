"""Tests for the database infrastructure layer, using an isolated in-memory
SQLite engine (never the app's configured database_url)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from campus_nav.infrastructure.database.models import (
    Base,
    NavigationEventRecord,
    NavigationSessionRecord,
    RouteRecord,
    SystemEvent,
    UserSession,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


def test_user_session_round_trip(db_session):
    session = UserSession()
    db_session.add(session)
    db_session.commit()

    fetched = db_session.get(UserSession, session.id)
    assert fetched is not None
    assert fetched.started_at is not None
    assert fetched.ended_at is None


def test_system_event_linked_to_session(db_session):
    session = UserSession()
    db_session.add(session)
    db_session.commit()

    event = SystemEvent(
        session_id=session.id, event_type="session_started", description="Test event."
    )
    db_session.add(event)
    db_session.commit()

    fetched = db_session.get(SystemEvent, event.id)
    assert fetched is not None
    assert fetched.session_id == session.id


def test_navigation_session_cascade_relationships(db_session):
    user_session = UserSession()
    db_session.add(user_session)
    db_session.commit()

    nav_session = NavigationSessionRecord(
        user_session_id=user_session.id,
        origin_node_id="main_gate",
        destination_node_id="library",
        status="navigating",
    )
    db_session.add(nav_session)
    db_session.commit()

    route = RouteRecord(
        navigation_session_id=nav_session.id,
        node_ids_json='["main_gate", "library"]',
        algorithm="astar",
        total_distance_meters=175.0,
    )
    event = NavigationEventRecord(
        navigation_session_id=nav_session.id,
        event_type="safety_alert",
        message="You have moved off the route.",
    )
    db_session.add_all([route, event])
    db_session.commit()

    fetched = db_session.get(NavigationSessionRecord, nav_session.id)
    assert len(fetched.routes) == 1
    assert len(fetched.events) == 1
    assert fetched.user_session.id == user_session.id


def test_deleting_user_session_cascades_to_navigation_sessions(db_session):
    user_session = UserSession()
    db_session.add(user_session)
    db_session.commit()

    nav_session = NavigationSessionRecord(
        user_session_id=user_session.id,
        origin_node_id="a",
        destination_node_id="b",
    )
    db_session.add(nav_session)
    db_session.commit()
    nav_session_id = nav_session.id

    db_session.delete(user_session)
    db_session.commit()

    assert db_session.get(NavigationSessionRecord, nav_session_id) is None
