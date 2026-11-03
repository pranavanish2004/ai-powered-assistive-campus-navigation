"""Tests for DecisionEngine -- the Phase 1.4d fusion orchestrator. Every
test uses MockMLLMProvider/KeywordKnowledgeBase, no real network access."""

import pytest

from campus_nav.assistant.context import (
    AssistantContext,
    NavigationContextSnapshot,
    VisionContextSnapshot,
)
from campus_nav.assistant.decision_engine import DecisionEngine
from campus_nav.core.exceptions import MLLMError, RAGError
from campus_nav.infrastructure.mllm.mock_mllm_provider import MockMLLMProvider
from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMResponse
from campus_nav.interfaces.rag_interface import KnowledgeDocument

pytestmark = pytest.mark.unit


def _kb_with_library_doc():
    kb = KeywordKnowledgeBase()
    kb.add_document(
        KnowledgeDocument(id="lib", title="Central Library", content="The main library on campus.")
    )
    return kb


def test_answer_returns_mllm_text():
    engine = DecisionEngine(rag_repository=_kb_with_library_doc(), mllm_provider=MockMLLMProvider())
    response = engine.answer("Where is the library?", AssistantContext())
    assert response.text == "This is a mock assistant response."


def test_answer_flags_used_rag_when_documents_retrieved():
    engine = DecisionEngine(rag_repository=_kb_with_library_doc(), mllm_provider=MockMLLMProvider())
    response = engine.answer("Where is the library?", AssistantContext())
    assert response.used_rag is True
    assert "lib" in response.source_document_ids


def test_answer_flags_used_rag_false_when_no_documents_match():
    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=MockMLLMProvider())
    response = engine.answer("completely unrelated gibberish query", AssistantContext())
    assert response.used_rag is False
    assert response.source_document_ids == []


def test_answer_flags_used_navigation_context():
    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=MockMLLMProvider())
    nav = NavigationContextSnapshot(state="navigating")
    response = engine.answer("How far?", AssistantContext(navigation=nav))
    assert response.used_navigation_context is True


def test_answer_flags_used_navigation_context_false_when_absent():
    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=MockMLLMProvider())
    response = engine.answer("How far?", AssistantContext())
    assert response.used_navigation_context is False


def test_answer_flags_used_vision_context():
    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=MockMLLMProvider())
    vision = VisionContextSnapshot(obstacles=[])
    response = engine.answer("What is in front of me?", AssistantContext(vision=vision))
    assert response.used_vision_context is True


def test_answer_flags_used_dashboard_only_when_query_relevant_and_data_present():
    from campus_nav.assistant.context import DashboardContextSnapshot

    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=MockMLLMProvider())
    dashboard = DashboardContextSnapshot()
    response = engine.answer("What's for lunch?", AssistantContext(dashboard=dashboard))
    assert response.used_dashboard is True


def test_answer_used_dashboard_false_for_irrelevant_query_even_with_data_present():
    from campus_nav.assistant.context import DashboardContextSnapshot

    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=MockMLLMProvider())
    dashboard = DashboardContextSnapshot()
    response = engine.answer("Where is the library?", AssistantContext(dashboard=dashboard))
    assert response.used_dashboard is False


def test_mllm_receives_deterministically_assembled_context():
    mock_mllm = MockMLLMProvider()
    engine = DecisionEngine(rag_repository=_kb_with_library_doc(), mllm_provider=mock_mllm)
    engine.answer("Where is the library?", AssistantContext())

    final_message = mock_mllm.last_messages[-1]
    assert "Central Library" in final_message.content
    assert "Where is the library?" in final_message.content


def test_mllm_failure_falls_back_to_honest_message_not_a_crash():
    class FailingMLLM:
        def generate(self, messages, max_tokens=512):
            raise MLLMError("simulated provider failure")

    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=FailingMLLM())
    response = engine.answer("Where is the library?", AssistantContext())

    assert "trouble answering" in response.text.lower()
    assert "navigation and safety" in response.text.lower()


def test_rag_failure_does_not_prevent_mllm_from_answering():
    class FailingRAG:
        def search(self, query, top_k=3):
            raise RAGError("simulated knowledge base failure")

        def add_document(self, document):
            pass

        def document_count(self):
            return 0

    engine = DecisionEngine(rag_repository=FailingRAG(), mllm_provider=MockMLLMProvider())
    response = engine.answer("Where is the library?", AssistantContext())

    assert response.text == "This is a mock assistant response."
    assert response.used_rag is False


def test_answer_never_raises_even_when_both_rag_and_mllm_fail():
    class FailingRAG:
        def search(self, query, top_k=3):
            raise RAGError("boom")

        def add_document(self, document):
            pass

        def document_count(self):
            return 0

    class FailingMLLM:
        def generate(self, messages, max_tokens=512):
            raise MLLMError("boom")

    engine = DecisionEngine(rag_repository=FailingRAG(), mllm_provider=FailingMLLM())
    response = engine.answer("Anything", AssistantContext())  # must not raise
    assert isinstance(response.text, str)
    assert len(response.text) > 0


def test_rag_top_k_is_respected():
    kb = KeywordKnowledgeBase()
    for i in range(5):
        kb.add_document(
            KnowledgeDocument(id=f"d{i}", title=f"Building {i}", content="campus building")
        )

    engine = DecisionEngine(rag_repository=kb, mllm_provider=MockMLLMProvider(), rag_top_k=2)
    response = engine.answer("campus building", AssistantContext())
    assert len(response.source_document_ids) == 2


def test_mllm_max_tokens_is_passed_through():
    class RecordingMLLM:
        def __init__(self):
            self.received_max_tokens = None

        def generate(self, messages, max_tokens=512):
            self.received_max_tokens = max_tokens
            return MLLMResponse(text="ok", model="test")

    mllm = RecordingMLLM()
    engine = DecisionEngine(
        rag_repository=KeywordKnowledgeBase(), mllm_provider=mllm, mllm_max_tokens=128
    )
    engine.answer("test", AssistantContext())
    assert mllm.received_max_tokens == 128


def test_knowledge_document_count_property():
    kb = _kb_with_library_doc()
    engine = DecisionEngine(rag_repository=kb, mllm_provider=MockMLLMProvider())
    assert engine.knowledge_document_count == 1


def test_conversation_history_is_forwarded_to_mllm():
    mock_mllm = MockMLLMProvider()
    engine = DecisionEngine(rag_repository=KeywordKnowledgeBase(), mllm_provider=mock_mllm)
    history = [
        MLLMMessage(role="user", content="Hi"),
        MLLMMessage(role="assistant", content="Hello!"),
    ]

    engine.answer("Follow-up", AssistantContext(conversation_history=history))

    assert history[0] in mock_mllm.last_messages
    assert history[1] in mock_mllm.last_messages
