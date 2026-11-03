"""Tests for assistant.prompt_builder -- the deterministic context
assembly that keeps every fusion decision inspectable and testable
without needing a real MLLM call."""

import pytest

from campus_nav.assistant.context import (
    AssistantContext,
    DashboardContextSnapshot,
    NavigationContextSnapshot,
    VisionContextSnapshot,
)
from campus_nav.assistant.prompt_builder import (
    build_context_block,
    build_messages,
    format_dashboard_context,
    format_navigation_context,
    format_vision_context,
    is_dashboard_relevant,
)
from campus_nav.interfaces.dashboard_interface import BusSchedule, MessMenu
from campus_nav.interfaces.mllm_interface import MLLMMessage
from campus_nav.interfaces.obstacle_interface import ObstacleEvent, ObstaclePosition
from campus_nav.interfaces.rag_interface import KnowledgeDocument, RetrievalResult
from campus_nav.vision.detection.categories import ObjectCategory

pytestmark = pytest.mark.unit


# --- is_dashboard_relevant ---


@pytest.mark.parametrize(
    "query",
    [
        "What's for lunch?",
        "Is there a bus?",
        "When is the next shuttle?",
        "What's on the mess menu today?",
    ],
)
def test_dashboard_relevant_queries_detected(query):
    assert is_dashboard_relevant(query) is True


@pytest.mark.parametrize(
    "query", ["Where is the library?", "What is in front of me?", "Read this sign."]
)
def test_non_dashboard_queries_not_flagged(query):
    assert is_dashboard_relevant(query) is False


# --- format_navigation_context ---


def test_format_navigation_context_no_session():
    context = AssistantContext()
    assert "No active navigation session" in format_navigation_context(context)


def test_format_navigation_context_active_session():
    nav = NavigationContextSnapshot(
        state="navigating",
        destination_node_id="library",
        current_node_id="admin_block",
        distance_remaining_meters=85.4,
        next_instruction_text="Continue straight.",
    )
    context = AssistantContext(navigation=nav)
    text = format_navigation_context(context)
    assert "navigating" in text
    assert "library" in text
    assert "85 meters" in text
    assert "Continue straight." in text


# --- format_vision_context ---


def test_format_vision_context_none():
    assert "No current camera information" in format_vision_context(AssistantContext())


def test_format_vision_context_with_announceable_obstacle():
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
    text = format_vision_context(AssistantContext(vision=vision))
    assert "stairs" in text.lower()


def test_format_vision_context_silent_obstacles_excluded():
    silent_obstacle = ObstacleEvent(
        track_id=1,
        category=ObjectCategory.BENCH,
        label="bench",
        position=ObstaclePosition.BESIDE_PATH,
        distance_meters=10.0,
        confidence=0.9,
        announce_text=None,
    )
    vision = VisionContextSnapshot(obstacles=[silent_obstacle])
    text = format_vision_context(AssistantContext(vision=vision))
    assert "bench" not in text.lower()


def test_format_vision_context_includes_ocr_text():
    vision = VisionContextSnapshot(ocr_texts=["Central Library"])
    text = format_vision_context(AssistantContext(vision=vision))
    assert "Central Library" in text


# --- format_dashboard_context ---


def test_format_dashboard_context_not_requested():
    assert "not requested" in format_dashboard_context(AssistantContext())


def test_format_dashboard_context_unavailable():
    context = AssistantContext(dashboard=DashboardContextSnapshot())
    assert "unavailable" in format_dashboard_context(context)


def test_format_dashboard_context_with_bus_schedule():
    dashboard = DashboardContextSnapshot(
        bus_schedule=BusSchedule(routes={"Gate to Hostel": ["08:00:00"]})
    )
    text = format_dashboard_context(AssistantContext(dashboard=dashboard))
    assert "Gate to Hostel" in text
    assert "08:00:00" in text


def test_format_dashboard_context_with_mess_menu():
    dashboard = DashboardContextSnapshot(
        mess_menu=MessMenu(days={"Monday": {"Breakfast": ["Idli"]}})
    )
    text = format_dashboard_context(AssistantContext(dashboard=dashboard))
    assert "Monday" in text
    assert "Idli" in text


# --- build_context_block / build_messages ---


def test_build_context_block_includes_all_sections():
    block = build_context_block(AssistantContext(), [])
    assert "Campus knowledge:" in block
    assert "Navigation status:" in block
    assert "Camera view:" in block
    assert "Dashboard:" in block


def test_build_messages_starts_with_system_message():
    messages = build_messages("Where is the library?", AssistantContext(), [])
    assert messages[0].role == "system"


def test_build_messages_ends_with_user_message_containing_query():
    messages = build_messages("Where is the library?", AssistantContext(), [])
    assert messages[-1].role == "user"
    assert "Where is the library?" in messages[-1].content


def test_build_messages_includes_conversation_history_in_order():
    history = [
        MLLMMessage(role="user", content="Hi"),
        MLLMMessage(role="assistant", content="Hello!"),
    ]
    context = AssistantContext(conversation_history=history)
    messages = build_messages("Follow-up question", context, [])

    assert messages[1] == history[0]
    assert messages[2] == history[1]
    assert messages[-1].role == "user"


def test_build_messages_includes_retrieved_knowledge():
    doc = KnowledgeDocument(id="lib", title="Central Library", content="The main library.")
    results = [RetrievalResult(document=doc, score=1.0)]
    messages = build_messages("Where is the library?", AssistantContext(), results)
    assert "Central Library" in messages[-1].content
    assert "The main library." in messages[-1].content


def test_build_messages_with_history_ending_in_assistant_turn_stays_alternating():
    """Documents the expected caller invariant: conversation_history holds
    completed turns and should end with an assistant reply, since the
    current turn's user query is passed separately via `query`. Under
    that invariant, the final two messages are assistant -> user, never
    two consecutive user turns."""
    history = [
        MLLMMessage(role="user", content="Hi"),
        MLLMMessage(role="assistant", content="Hello!"),
    ]
    context = AssistantContext(conversation_history=history)
    messages = build_messages("Follow-up question", context, [])

    assert messages[-2].role == "assistant"
    assert messages[-1].role == "user"
