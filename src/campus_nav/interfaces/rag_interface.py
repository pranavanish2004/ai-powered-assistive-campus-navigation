"""
Retrieval-Augmented Generation (RAG) abstraction.

Design rationale
-----------------
This is the assistant's grounding mechanism: campus facts (building
descriptions, accessibility info, emergency contacts, campus rules) are
retrieved as verbatim documents and handed to the MLLM as context, rather
than the MLLM being expected to "know" campus-specific facts on its own.
This is what keeps the assistant honest -- per the project's AI Behaviour
requirements, it must never invent a building location or an accessible
entrance it has no actual record of.

Two backends exist (see infrastructure/rag/): a pure-Python keyword
retriever (the safe default, no ML dependencies) and an optional
embedding-based retriever for genuine semantic search. Both implement this
same interface, so the Decision Engine never knows or cares which is
active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class KnowledgeCategory(str, Enum):
    """Matches the categories the original specification named for RAG:
    buildings, departments, hostels, bus routes, mess menus, faculty
    offices, accessibility information, emergency contacts, campus rules,
    medical centre, announcements, events."""

    BUILDING = "building"
    DEPARTMENT = "department"
    HOSTEL = "hostel"
    ACCESSIBILITY = "accessibility"
    EMERGENCY = "emergency"
    RULES = "rules"
    MEDICAL = "medical"
    GENERAL = "general"


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """A single retrievable fact or short passage."""

    id: str
    title: str
    content: str
    category: KnowledgeCategory = KnowledgeCategory.GENERAL
    metadata: dict = field(default_factory=dict)
    """Free-form extra context, e.g. {"node_id": "library"} linking a
    document back to a specific Campus Graph node."""
    source: str = "unknown"
    """Where this document came from, e.g. "campus_graph",
    "static_knowledge", "geojson_import:campus.geojson",
    "csv_import:buildings.csv" -- lets a caller (or a future UI) show
    provenance, and lets the ingestion pipeline (Milestone 1.4-refinement)
    trace every document back to its originating dataset."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One retrieved document plus its relevance score for a given query."""

    document: KnowledgeDocument
    score: float
    """Backend-specific relevance score, higher = more relevant. Not
    normalized to a fixed range across backends -- callers should treat it
    as a ranking signal, not an absolute confidence value."""
    retrieval_method: str = "unknown"
    """Which backend/algorithm produced this result, e.g. "keyword",
    "tfidf", "chroma", "faiss" -- lets a caller distinguish, in a mixed or
    evolving deployment, how a given fact was found. Each
    RAGRepositoryInterface implementation MUST set this explicitly rather
    than leaving the default, since "unknown" defeats the purpose."""


class RAGRepositoryInterface(ABC):
    """Abstract base class for the campus knowledge base."""

    @abstractmethod
    def add_document(self, document: KnowledgeDocument) -> None:
        """Add or replace (by id) a document in the knowledge base."""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return up to `top_k` documents most relevant to `query`, ordered
        by descending score. Returns an empty list (not an error) if the
        knowledge base has no relevant -- or any -- documents; "nothing
        matched" is normal, routine control flow."""

    @abstractmethod
    def document_count(self) -> int:
        """Number of documents currently indexed. Used by the
        `/assistant/status` endpoint and tests, not by any retrieval
        logic itself."""
