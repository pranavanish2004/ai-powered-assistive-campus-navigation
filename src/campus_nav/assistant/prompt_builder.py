"""
Prompt assembly for the Decision Engine.

Design rationale
-----------------
Every decision about *what information is relevant* is made here, in
plain, unit-testable Python — never by asking the MLLM to decide what to
fetch (no tool-calling/function-calling loop). This matters for three
reasons specific to this project:

1. **The safety boundary is enforced by code, not by prompt wording.**
   The MLLM never gets the opportunity to "decide" to reroute someone or
   invent a distance; it receives already-computed, already-verified
   facts and is asked only to phrase them naturally.
2. **Determinism is testable.** `is_dashboard_relevant("what's for lunch")`
   either returns True or it doesn't — a unit test can assert that
   directly. Whether an LLM would have "decided" to call a mess-menu tool
   is not something a test suite can assert reliably.
3. **Grounding is explicit.** Retrieved knowledge is placed in the prompt
   with clear structure and an explicit instruction not to state anything
   as fact beyond what's provided — directly implementing the project's
   "the assistant must never invent a building location" requirement.
"""

from __future__ import annotations

from campus_nav.assistant.context import AssistantContext
from campus_nav.interfaces.mllm_interface import MLLMMessage
from campus_nav.interfaces.rag_interface import RetrievalResult

_SYSTEM_PROMPT = """You are the AI Campus Navigation Assistant, a voice-first mobility aid \
for a visually impaired student at IIT Hyderabad.

Rules you must always follow:
- You do not control navigation, routing, obstacle detection, or safety alerts. Those are \
handled by separate, deterministic systems. Never claim to start, stop, or change navigation \
yourself, and never invent turn-by-turn directions.
- Only state campus facts (building locations, accessibility details, schedules) that are \
explicitly provided to you in the Context section below. If something is not in the Context, \
say you don't have that information rather than guessing.
- Speak naturally and concisely, the way a helpful human companion would -- not like a system \
reciting sensor output. For example, prefer "There is a bicycle parked ahead, walk slightly \
left" over "Obstacle detected."
- Keep responses short enough to be comfortable to listen to, not read.
"""

_DASHBOARD_KEYWORDS = frozenset(
    {
        "bus",
        "shuttle",
        "transport",
        "ev",
        "mess",
        "menu",
        "food",
        "lunch",
        "dinner",
        "breakfast",
        "snacks",
        "canteen",
    }
)


def is_dashboard_relevant(query: str) -> bool:
    """Deterministic keyword check deciding whether dashboard data (bus
    schedule, mess menu) should be fetched and included for this query.
    Intentionally simple and inspectable rather than left to model
    judgment -- see module docstring."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in _DASHBOARD_KEYWORDS)


def format_retrieved_knowledge(results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant campus knowledge was found for this query."
    lines = [f"- {r.document.title}: {r.document.content}" for r in results]
    return "\n".join(lines)


def format_navigation_context(context: AssistantContext) -> str:
    nav = context.navigation
    if nav is None:
        return "No active navigation session."

    parts = [f"Navigation state: {nav.state}."]
    if nav.destination_node_id:
        parts.append(f"Destination: {nav.destination_node_id}.")
    if nav.current_node_id:
        parts.append(f"Current position (nearest known point): {nav.current_node_id}.")
    if nav.distance_remaining_meters is not None:
        parts.append(f"Distance remaining: {round(nav.distance_remaining_meters)} meters.")
    if nav.next_instruction_text:
        parts.append(f"Next planned instruction: {nav.next_instruction_text}")
    return " ".join(parts)


def format_vision_context(context: AssistantContext) -> str:
    vision = context.vision
    if vision is None or (not vision.obstacles and not vision.ocr_texts):
        return "No current camera information available."

    parts = []
    announceable = [o for o in vision.obstacles if o.announce_text]
    if announceable:
        descriptions = ", ".join(f"{o.label} ({o.position.value})" for o in announceable)
        parts.append(f"Currently visible obstacles: {descriptions}.")
    if vision.ocr_texts:
        parts.append(f"Text visible in camera view: {'; '.join(vision.ocr_texts)}.")
    return " ".join(parts) if parts else "No current camera information available."


def format_dashboard_context(context: AssistantContext) -> str:
    dashboard = context.dashboard
    if dashboard is None:
        return "Dashboard data was not requested for this query."

    parts = []
    if dashboard.bus_schedule is not None:
        route_summary = ", ".join(
            f"{route}: {', '.join(times)}" for route, times in dashboard.bus_schedule.routes.items()
        )
        parts.append(
            f"Bus schedule -- {route_summary}."
            if route_summary
            else "Bus schedule: no routes listed."
        )
    if dashboard.mess_menu is not None:
        meal_summaries = []
        for day, meals in dashboard.mess_menu.days.items():
            meal_parts = [f"{meal} ({', '.join(items)})" for meal, items in meals.items()]
            meal_summaries.append(f"{day}: {', '.join(meal_parts)}")
        day_summary = "; ".join(meal_summaries)
        parts.append(
            f"Mess menu -- {day_summary}." if day_summary else "Mess menu: no data listed."
        )
    if not parts:
        return "Dashboard data was requested but is currently unavailable."
    return " ".join(parts)


def build_context_block(context: AssistantContext, retrieval_results: list[RetrievalResult]) -> str:
    """Assemble the full, structured context block placed alongside the
    user's question in the final prompt."""
    return (
        "Campus knowledge:\n"
        f"{format_retrieved_knowledge(retrieval_results)}\n\n"
        "Navigation status:\n"
        f"{format_navigation_context(context)}\n\n"
        "Camera view:\n"
        f"{format_vision_context(context)}\n\n"
        "Dashboard:\n"
        f"{format_dashboard_context(context)}"
    )


def build_messages(
    query: str, context: AssistantContext, retrieval_results: list[RetrievalResult]
) -> list[MLLMMessage]:
    """Assemble the full message list sent to the MLLM provider: one
    system message establishing rules and persona, prior conversation
    history, and a final user message combining the assembled context
    block with the actual question -- kept as a single combined user
    message (rather than a separate context message) so providers that
    expect strictly alternating user/assistant turns are never given two
    consecutive user-role messages.

    Caller invariant: `context.conversation_history` must hold only
    *completed* turns (ending with an assistant reply, if non-empty) --
    the current turn's user query is passed separately via `query`, not
    appended to history by this function. A history that itself ends with
    an unanswered user turn would produce two consecutive user messages,
    which this function does not guard against; that is a caller bug to
    fix at the history-construction site, not something to paper over here.
    """
    context_block = build_context_block(context, retrieval_results)
    final_user_message = MLLMMessage(
        role="user",
        content=f"Context:\n{context_block}\n\nQuestion: {query}",
    )

    return [
        MLLMMessage(role="system", content=_SYSTEM_PROMPT),
        *context.conversation_history,
        final_user_message,
    ]
