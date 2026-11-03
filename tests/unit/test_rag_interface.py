"""Tests for the RAG interface contract and KeywordKnowledgeBase."""

import pytest

from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.interfaces.rag_interface import (
    KnowledgeCategory,
    KnowledgeDocument,
    RAGRepositoryInterface,
)

pytestmark = pytest.mark.unit


def test_cannot_instantiate_rag_interface_directly():
    with pytest.raises(TypeError):
        RAGRepositoryInterface()  # type: ignore[abstract]


def test_empty_knowledge_base_returns_no_results():
    kb = KeywordKnowledgeBase()
    assert kb.search("library") == []
    assert kb.document_count() == 0


def test_add_and_retrieve_exact_title_match():
    kb = KeywordKnowledgeBase()
    doc = KnowledgeDocument(id="d1", title="Central Library", content="The main library on campus.")
    kb.add_document(doc)

    results = kb.search("library")
    assert len(results) == 1
    assert results[0].document.id == "d1"


def test_search_ranks_more_relevant_documents_higher():
    kb = KeywordKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="lib", title="Central Library", content="Books and study space.")
    )
    kb.add_document(
        KnowledgeDocument(id="med", title="Medical Centre", content="First aid and consultation.")
    )

    results = kb.search("library books")
    assert results[0].document.id == "lib"


def test_search_returns_no_results_for_completely_unrelated_query():
    kb = KeywordKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="lib", title="Central Library", content="Books and study space.")
    )

    assert kb.search("xyzzy quibblewhat") == []


def test_top_k_limits_result_count():
    kb = KeywordKnowledgeBase()
    for i in range(10):
        kb.add_document(
            KnowledgeDocument(
                id=f"doc{i}", title=f"Building {i}", content="campus building location"
            )
        )
    results = kb.search("campus building", top_k=3)
    assert len(results) == 3


def test_add_document_with_same_id_replaces_existing():
    kb = KeywordKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="d1", title="Old Title", content="old content"))
    kb.add_document(KnowledgeDocument(id="d1", title="New Title", content="new content"))

    assert kb.document_count() == 1
    results = kb.search("new content")
    assert results[0].document.title == "New Title"


def test_document_count_reflects_added_documents():
    kb = KeywordKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="A", content="a"))
    kb.add_document(KnowledgeDocument(id="b", title="B", content="b"))
    assert kb.document_count() == 2


def test_empty_query_returns_no_results():
    kb = KeywordKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="A", content="something"))
    assert kb.search("") == []


def test_knowledge_document_defaults():
    doc = KnowledgeDocument(id="x", title="X", content="content")
    assert doc.category is KnowledgeCategory.GENERAL
    assert doc.metadata == {}


def test_case_insensitive_matching():
    kb = KeywordKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="lib", title="Central Library", content="Books."))
    results_upper = kb.search("LIBRARY")
    results_lower = kb.search("library")
    assert len(results_upper) == 1
    assert results_upper[0].score == results_lower[0].score


def test_stopword_only_overlap_does_not_produce_a_spurious_match():
    """Regression test matching TFIDFKnowledgeBase's equivalent: a query
    whose only overlap with a document is common function words must not
    surface that document as relevant."""
    kb = KeywordKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="sncc", title="SNCC", content="Large open hall used for events.")
    )
    results = kb.search("what's for lunch")
    assert results == []
