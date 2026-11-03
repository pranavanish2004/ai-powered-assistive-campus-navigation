"""Tests for GroundedMockMLLMProvider -- verifies it never returns the old
fixed placeholder string and genuinely synthesizes responses from the
structured context block prompt_builder assembles."""

import pytest

from campus_nav.assistant.context import (
    AssistantContext,
    DashboardContextSnapshot,
    NavigationContextSnapshot,
    VisionContextSnapshot,
)
from campus_nav.assistant.prompt_builder import build_messages
from campus_nav.infrastructure.mllm.grounded_mock_provider import GroundedMockMLLMProvider
from campus_nav.interfaces.dashboard_interface import BusSchedule, MessMenu
from campus_nav.interfaces.obstacle_interface import ObstacleEvent, ObstaclePosition
from campus_nav.interfaces.rag_interface import KnowledgeDocument, RetrievalResult
from campus_nav.vision.detection.categories import ObjectCategory

pytestmark = pytest.mark.unit

_OLD_PLACEHOLDER = "This is a mock assistant response."


def _messages_for(query, context=None, retrieval_results=None):
    return build_messages(query, context or AssistantContext(), retrieval_results or [])


def test_never_returns_old_fixed_placeholder():
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("Where is the library?"))
    assert response.text != _OLD_PLACEHOLDER


def test_empty_context_returns_honest_fallback_not_placeholder():
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("asdkjaslkdj random gibberish"))
    assert response.text != _OLD_PLACEHOLDER
    assert "don't have specific information" in response.text.lower()


def test_uses_top_retrieved_knowledge_fact():
    doc = KnowledgeDocument(
        id="lib", title="Central Library", content="The main library on campus."
    )
    results = [RetrievalResult(document=doc, score=1.0, retrieval_method="tfidf")]
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("Where is the library?", retrieval_results=results))
    assert "The main library on campus." in response.text


def test_uses_navigation_context_including_distance():
    nav = NavigationContextSnapshot(
        state="navigating",
        destination_node_id="biomedical",
        distance_remaining_meters=353.0,
        next_instruction_text="Head toward Administration Block.",
    )
    context = AssistantContext(navigation=nav)
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("How far do I have left?", context=context))
    assert "353 meters" in response.text
    assert "Head toward Administration Block." in response.text


def test_different_contexts_produce_different_responses():
    provider = GroundedMockMLLMProvider()
    doc = KnowledgeDocument(id="lib", title="Central Library", content="The main library.")
    results = [RetrievalResult(document=doc, score=1.0, retrieval_method="tfidf")]

    response_with_knowledge = provider.generate(
        _messages_for("Where is the library?", retrieval_results=results)
    )
    response_without_knowledge = provider.generate(_messages_for("Where is the library?"))

    assert response_with_knowledge.text != response_without_knowledge.text


def test_uses_dashboard_context_bus_schedule():
    dashboard = DashboardContextSnapshot(
        bus_schedule=BusSchedule(routes={"Gate to Hostel": ["08:00:00"]})
    )
    context = AssistantContext(dashboard=dashboard)
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("Is there a bus?", context=context))
    assert "Gate to Hostel" in response.text


def test_uses_dashboard_context_mess_menu():
    dashboard = DashboardContextSnapshot(mess_menu=MessMenu(days={"Monday": {"Lunch": ["Rice"]}}))
    context = AssistantContext(dashboard=dashboard)
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("What's for lunch?", context=context))
    assert "Monday" in response.text
    assert "Rice" in response.text


def test_uses_vision_context_announceable_obstacles():
    obstacle = ObstacleEvent(
        track_id=1,
        category=ObjectCategory.STAIRS,
        label="stairs",
        position=ObstaclePosition.BLOCKING,
        distance_meters=1.0,
        confidence=0.9,
        announce_text="Stairs ahead.",
    )
    vision = VisionContextSnapshot(obstacles=[obstacle])
    context = AssistantContext(vision=vision)
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("What is in front of me?", context=context))
    assert "stairs" in response.text.lower()


def test_records_last_messages_like_mock_provider():
    provider = GroundedMockMLLMProvider()
    messages = _messages_for("test query")
    provider.generate(messages)
    assert provider.last_messages == messages


def test_response_model_is_labeled_mock_grounded():
    provider = GroundedMockMLLMProvider()
    response = provider.generate(_messages_for("test"))
    assert response.model == "mock-grounded"


def test_handles_empty_messages_list_gracefully():
    provider = GroundedMockMLLMProvider()
    response = provider.generate([])
    assert response.text != _OLD_PLACEHOLDER
    assert isinstance(response.text, str)
