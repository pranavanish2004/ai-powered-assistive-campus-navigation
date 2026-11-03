"""RAG backend factory -- Settings-driven selection, mirroring
camera_factory.py / vision_factory.py / dashboard_factory.py."""

from __future__ import annotations

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.rag.campus_knowledge_loader import build_campus_knowledge_base
from campus_nav.infrastructure.rag.chroma_knowledge_base import ChromaKnowledgeBase
from campus_nav.infrastructure.rag.faiss_knowledge_base import FAISSKnowledgeBase
from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.infrastructure.rag.tfidf_knowledge_base import TFIDFKnowledgeBase
from campus_nav.interfaces.rag_interface import RAGRepositoryInterface
from campus_nav.navigation.graph.models import CampusGraph


def create_empty_knowledge_base(settings: Settings) -> RAGRepositoryInterface:
    if settings.rag_backend == "tfidf":
        return TFIDFKnowledgeBase()
    if settings.rag_backend == "chroma":
        return ChromaKnowledgeBase()
    if settings.rag_backend == "faiss":
        return FAISSKnowledgeBase()
    return KeywordKnowledgeBase()


def create_campus_knowledge_base(settings: Settings, graph: CampusGraph) -> RAGRepositoryInterface:
    """Build and populate a knowledge base from `graph` plus the static
    supplementary knowledge file configured in Settings."""
    repository = create_empty_knowledge_base(settings)
    return build_campus_knowledge_base(repository, graph, settings.rag_knowledge_path)
