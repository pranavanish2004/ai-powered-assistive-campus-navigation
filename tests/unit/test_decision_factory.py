"""Tests for the Decision Engine factory."""

import pytest

from campus_nav.assistant.decision_engine import DecisionEngine
from campus_nav.assistant.decision_factory import create_decision_engine
from campus_nav.config.settings import Settings

pytestmark = pytest.mark.unit


def test_create_decision_engine_returns_fully_wired_engine(pilot_graph):
    settings = Settings(_env_file=None, rag_knowledge_path="tests/fixtures/campus_knowledge.json")
    engine = create_decision_engine(settings, pilot_graph)
    assert isinstance(engine, DecisionEngine)
    assert engine.knowledge_document_count > 0


def test_create_decision_engine_uses_grounded_mock_mllm_by_default(pilot_graph):
    settings = Settings(_env_file=None, rag_knowledge_path="tests/fixtures/campus_knowledge.json")
    engine = create_decision_engine(settings, pilot_graph)
    from campus_nav.infrastructure.mllm.grounded_mock_provider import GroundedMockMLLMProvider

    assert isinstance(engine.mllm_provider, GroundedMockMLLMProvider)
