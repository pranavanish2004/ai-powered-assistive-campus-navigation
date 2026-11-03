"""Tests for the RAG backend factory."""

import pytest

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.rag.chroma_knowledge_base import ChromaKnowledgeBase
from campus_nav.infrastructure.rag.faiss_knowledge_base import FAISSKnowledgeBase
from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.infrastructure.rag.rag_factory import (
    create_campus_knowledge_base,
    create_empty_knowledge_base,
)
from campus_nav.infrastructure.rag.tfidf_knowledge_base import TFIDFKnowledgeBase

pytestmark = pytest.mark.unit


def test_default_backend_is_tfidf():
    settings = Settings(_env_file=None)
    kb = create_empty_knowledge_base(settings)
    assert isinstance(kb, TFIDFKnowledgeBase)


def test_keyword_backend_selected_returns_keyword_instance():
    settings = Settings(_env_file=None, rag_backend="keyword")
    kb = create_empty_knowledge_base(settings)
    assert isinstance(kb, KeywordKnowledgeBase)


def test_chroma_backend_selected_returns_chroma_instance():
    settings = Settings(_env_file=None, rag_backend="chroma")
    kb = create_empty_knowledge_base(settings)
    assert isinstance(kb, ChromaKnowledgeBase)
    # Not calling .search()/.add_document() here -- that would require the
    # optional [rag] extra (chromadb, sentence-transformers) to be
    # installed; see infrastructure adapter docstrings for the lazy-import
    # contract, matching YOLOv8Detector/WhisperSTT's pattern.


def test_faiss_backend_selected_returns_faiss_instance():
    settings = Settings(_env_file=None, rag_backend="faiss")
    kb = create_empty_knowledge_base(settings)
    assert isinstance(kb, FAISSKnowledgeBase)
    # Same rationale as Chroma above -- no .search() call here.


def test_create_campus_knowledge_base_populates_from_graph_and_static_file(pilot_graph):
    settings = Settings(_env_file=None, rag_knowledge_path="tests/fixtures/campus_knowledge.json")
    kb = create_campus_knowledge_base(settings, pilot_graph)
    assert kb.document_count() > pilot_graph.node_count()  # graph docs + static docs


def test_create_campus_knowledge_base_uses_tfidf_by_default(pilot_graph):
    settings = Settings(_env_file=None, rag_knowledge_path="tests/fixtures/campus_knowledge.json")
    kb = create_campus_knowledge_base(settings, pilot_graph)
    assert isinstance(kb, TFIDFKnowledgeBase)
