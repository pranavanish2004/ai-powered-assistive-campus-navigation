"""Tests for FAISSKnowledgeBase's lazy-import contract -- faiss and
sentence-transformers are not part of the default install (see the
[rag] optional extra), so these tests verify the "not installed" path
raises a clear ModelLoadError, matching every other optional-ML-backend
adapter in this codebase (YOLOv8Detector, EasyOCRService, MiDaS, Chroma)."""

import sys
from unittest.mock import patch

import pytest

from campus_nav.core.exceptions import ModelLoadError
from campus_nav.infrastructure.rag.faiss_knowledge_base import FAISSKnowledgeBase
from campus_nav.interfaces.rag_interface import KnowledgeDocument

pytestmark = pytest.mark.unit


def test_document_count_and_add_document_require_no_dependencies():
    """add_document/document_count must work with zero ML dependencies --
    only search() (which needs the embedding model and FAISS index)
    requires the optional extra."""
    kb = FAISSKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="A", content="test"))
    assert kb.document_count() == 1


def test_search_raises_model_load_error_when_sentence_transformers_not_installed():
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        kb = FAISSKnowledgeBase()
        kb.add_document(KnowledgeDocument(id="a", title="A", content="test"))
        with pytest.raises(ModelLoadError):
            kb.search("test")


def test_search_on_empty_knowledge_base_returns_empty_without_needing_dependencies():
    """No documents means no index to build -- must short-circuit before
    ever touching faiss/sentence-transformers imports."""
    kb = FAISSKnowledgeBase()
    assert kb.search("anything") == []
