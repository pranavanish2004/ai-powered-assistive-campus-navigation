"""
ORM models.

Design rationale
-----------------
Phase 1.1 established the minimal session/event infrastructure. Phase 1.2
adds the tables the Navigation Engine needs to persist active routes and
navigation-specific events, *without* persisting the campus graph itself —
the graph is authored/versioned data (see `navigation.graph`), loaded from a
JSON file via the graph repository, not something users mutate through the
app. Only per-user navigation *state* belongs in the database.

- `UserSession`: a continuous period of assistant use by one user. Later
  modules (Navigation Engine, Conversation Manager) foreign-key against it.
- `SystemEvent`: durable, queryable record of significant events (session
  lifecycle, safety alerts, errors) — distinct from rotating log files,
  which are for developers grepping locally, not for building dashboards.
- `NavigationSessionRecord`: one active-or-completed navigation attempt
  (origin → destination) within a UserSession.
- `RouteRecord`: the computed route for a NavigationSessionRecord, stored as
  an ordered list of node ids (JSON) plus summary metrics. Kept separate
  from NavigationSessionRecord so a reroute can create a new RouteRecord
  without losing the history of previous routes for that session.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models in the application."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ===========================================================================
# Phase 1.1 — session & event infrastructure
# ===========================================================================
class UserSession(Base):
    """A single continuous period of assistant use by one user."""

    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    navigation_sessions: Mapped[list[NavigationSessionRecord]] = relationship(
        back_populates="user_session", cascade="all, delete-orphan"
    )


class SystemEvent(Base):
    """A significant, durable event worth querying later (session
    lifecycle, safety alerts, errors) — separate from rotating log files."""

    __tablename__ = "system_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ===========================================================================
# Phase 1.2 — navigation state
# ===========================================================================
class NavigationSessionRecord(Base):
    """One navigation attempt (origin -> destination) within a UserSession.
    `status` mirrors campus_nav.navigation.engine.state_machine.NavState
    values, stored as a plain string so the DB schema doesn't need to change
    if the state machine's enum evolves."""

    __tablename__ = "navigation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("user_sessions.id"), nullable=False
    )
    origin_node_id: Mapped[str] = mapped_column(String(64))
    destination_node_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="idle")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user_session: Mapped[UserSession] = relationship(back_populates="navigation_sessions")
    routes: Mapped[list[RouteRecord]] = relationship(
        back_populates="navigation_session", cascade="all, delete-orphan"
    )
    events: Mapped[list[NavigationEventRecord]] = relationship(
        back_populates="navigation_session", cascade="all, delete-orphan"
    )


class RouteRecord(Base):
    """A computed route for a navigation session. Kept separate from
    NavigationSessionRecord (rather than overwriting it) so a reroute
    preserves the history of previously computed routes."""

    __tablename__ = "routes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    navigation_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("navigation_sessions.id"), nullable=False
    )
    node_ids_json: Mapped[str] = mapped_column(Text)  # JSON list of node ids, in order
    algorithm: Mapped[str] = mapped_column(String(32))  # "dijkstra" | "astar"
    total_distance_meters: Mapped[float] = mapped_column(Float)
    accessible_only: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    navigation_session: Mapped[NavigationSessionRecord] = relationship(back_populates="routes")


class NavigationEventRecord(Base):
    """Navigation-specific durable event: off-route warning, reroute,
    arrival, safety alert. Distinct from the generic SystemEvent table so
    navigation analytics don't have to filter a shared table by prefix."""

    __tablename__ = "navigation_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    navigation_session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("navigation_sessions.id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
    node_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    navigation_session: Mapped[NavigationSessionRecord] = relationship(back_populates="events")
