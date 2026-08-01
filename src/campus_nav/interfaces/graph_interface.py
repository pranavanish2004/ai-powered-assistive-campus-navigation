"""
Campus graph repository abstraction.

The campus graph is authored/versioned data (see navigation/graph/models.py
docstring) — this interface is the Repository Pattern boundary between
"where the graph data lives" (a JSON file today; potentially a database or
a remote config service later) and the routing/navigation code that
consumes it. Routing code should depend on a loaded `CampusGraph` object,
never on this interface directly; this interface exists for the
application-startup code that decides *how* to obtain that object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from campus_nav.navigation.graph.models import CampusGraph


class GraphRepositoryInterface(ABC):
    """Abstract base class for loading/persisting the campus graph."""

    @abstractmethod
    def load(self) -> CampusGraph:
        """Load and return the campus graph. Implementations should
        validate the graph (see CampusGraph.validate()) before returning it
        — routing over an unvalidated graph could silently produce an
        unsafe route."""

    @abstractmethod
    def save(self, graph: CampusGraph) -> None:
        """Persist `graph`. Used by graph-authoring tools, not by the
        runtime navigation path."""
