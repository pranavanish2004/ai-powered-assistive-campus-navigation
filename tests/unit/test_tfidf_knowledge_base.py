"""Tests for TFIDFKnowledgeBase -- real term-frequency/inverse-document-
frequency retrieval with cosine similarity."""

import pytest

from campus_nav.infrastructure.rag.tfidf_knowledge_base import TFIDFKnowledgeBase
from campus_nav.interfaces.rag_interface import KnowledgeDocument

pytestmark = pytest.mark.unit


def test_empty_knowledge_base_returns_no_results():
    kb = TFIDFKnowledgeBase()
    assert kb.search("library") == []
    assert kb.document_count() == 0


def test_exact_title_match_ranks_first():
    kb = TFIDFKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="lib", title="Central Library", content="Books and study space.")
    )
    kb.add_document(
        KnowledgeDocument(
            id="canteen", title="Student Canteen", content="Food and snacks near the Library."
        )
    )

    results = kb.search("library")
    assert results[0].document.id == "lib"


def test_rare_distinctive_term_outweighs_common_term():
    """TF-IDF's core advantage over plain keyword overlap: a distinctive
    term appearing in only one document should dominate ranking even if
    a common term appears in many documents."""
    kb = TFIDFKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="a", title="Building A", content="campus building location")
    )
    kb.add_document(
        KnowledgeDocument(id="b", title="Building B", content="campus building location")
    )
    kb.add_document(
        KnowledgeDocument(
            id="bio",
            title="Biomedical Engineering",
            content="campus building for biomedical research",
        )
    )

    results = kb.search("biomedical")
    assert results[0].document.id == "bio"


def test_results_tagged_with_tfidf_retrieval_method():
    kb = TFIDFKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="A", content="test content"))
    results = kb.search("test")
    assert results[0].retrieval_method == "tfidf"


def test_unrelated_query_returns_no_results():
    kb = TFIDFKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="Library", content="Books."))
    assert kb.search("xyzzy quibblewhat") == []


def test_empty_query_returns_no_results():
    kb = TFIDFKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="A", content="something"))
    assert kb.search("") == []


def test_top_k_limits_result_count():
    kb = TFIDFKnowledgeBase()
    for i in range(10):
        kb.add_document(
            KnowledgeDocument(id=f"doc{i}", title=f"Building {i}", content="campus building")
        )
    results = kb.search("campus building", top_k=3)
    assert len(results) == 3


def test_add_document_after_search_invalidates_and_rebuilds_index():
    kb = TFIDFKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="Library", content="Books."))
    kb.search("library")  # builds initial index

    kb.add_document(KnowledgeDocument(id="b", title="Medical Centre", content="First aid."))
    results = kb.search("medical")
    assert results[0].document.id == "b"


def test_document_count_reflects_added_documents():
    kb = TFIDFKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="a", title="A", content="a"))
    kb.add_document(KnowledgeDocument(id="b", title="B", content="b"))
    assert kb.document_count() == 2


def test_document_with_empty_content_does_not_crash_search():
    kb = TFIDFKnowledgeBase()
    kb.add_document(KnowledgeDocument(id="empty", title="", content=""))
    kb.add_document(KnowledgeDocument(id="real", title="Library", content="Books."))
    results = kb.search("library")
    assert any(r.document.id == "real" for r in results)


def test_scores_are_sorted_descending():
    kb = TFIDFKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="a", title="Library Building", content="library library library")
    )
    kb.add_document(
        KnowledgeDocument(id="b", title="Other Building", content="mentions library once")
    )
    results = kb.search("library")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_stopword_only_overlap_does_not_produce_a_spurious_match():
    """Regression test: a query whose only overlap with a document is
    common function words (e.g. "for") must not surface that document as
    if it were relevant."""
    kb = TFIDFKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(
            id="sncc", title="SNCC", content="Large open hall used for events near campus."
        )
    )
    kb.add_document(
        KnowledgeDocument(
            id="canteen",
            title="Student Canteen",
            content="Offers snacks and meals throughout the day.",
        )
    )

    results = kb.search("what's for lunch")
    assert results == []
