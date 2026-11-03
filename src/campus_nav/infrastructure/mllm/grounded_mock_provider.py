"""
Grounded mock MLLM provider -- the production default for `mllm_provider =
"mock"`.

Why this exists
-----------------
`MockMLLMProvider` (mock_mllm_provider.py) is a pure test double: it
returns a fixed string regardless of input, which is exactly right for
unit-testing DecisionEngine's fusion *logic* in isolation, but wrong as
the actual runtime default -- a real deployment with no API key
configured must still produce meaningful, grounded answers, not a
placeholder. This class is that default: a deterministic, template-based
"phrasing" engine that reads the exact context block `prompt_builder`
assembles and synthesizes a natural sentence from it, with zero network
access and zero API keys required.

It deliberately parses the same structured text block a real MLLM
provider would receive (rather than requiring a separate, richer
structured-context argument), because `MLLMProviderInterface.generate()`
must stay a plain messages-in/text-out contract every provider -- real or
mock -- implements identically. `prompt_builder.build_context_block()`'s
section headers ("Campus knowledge:", "Navigation status:", etc.) are the
stable parsing anchors; both modules live in this codebase and evolve
together, so this is an internal contract, not a fragile guess against an
external format.
"""

from __future__ import annotations

import re

from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMProviderInterface, MLLMResponse

_NO_KNOWLEDGE_MARKER = "No relevant campus knowledge was found for this query."
_NO_NAV_MARKER = "No active navigation session."
_NO_CAMERA_MARKER = "No current camera information available."
_NO_DASHBOARD_MARKERS = (
    "Dashboard data was not requested for this query.",
    "Dashboard data was requested but is currently unavailable.",
)

_FALLBACK_TEXT = (
    "I don't have specific information about that yet. I can help with campus "
    "navigation, obstacles around you, and reading nearby signs -- try asking "
    "one of those."
)

_SECTION_ORDER = (
    "Campus knowledge:",
    "Navigation status:",
    "Camera view:",
    "Dashboard:",
    "Question:",
)


def _extract_section(context_block: str, header: str) -> str:
    """Extract the text between `header` and whichever known section
    header comes next (or end of string), matching the exact structure
    `prompt_builder.build_context_block` always produces."""
    try:
        start = context_block.index(header) + len(header)
    except ValueError:
        return ""

    remainder = context_block[start:]
    next_positions = [remainder.index(h) for h in _SECTION_ORDER if h != header and h in remainder]
    end = min(next_positions) if next_positions else len(remainder)
    return remainder[:end].strip()


def _first_knowledge_fact(knowledge_section: str) -> str:
    """`format_retrieved_knowledge` renders each result as `- Title:
    content`; take the first (highest-scored) fact's content as the
    primary grounded sentence."""
    first_line = knowledge_section.splitlines()[0] if knowledge_section else ""
    match = re.match(r"^-\s*[^:]+:\s*(.+)$", first_line)
    return match.group(1).strip() if match else knowledge_section.strip()


class GroundedMockMLLMProvider(MLLMProviderInterface):
    """Deterministically synthesizes a grounded response from the
    assembled context block -- no API key, no network access, never a
    fixed placeholder string."""

    def __init__(self) -> None:
        self.last_messages: list[MLLMMessage] = []

    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        self.last_messages = list(messages)
        final_message = messages[-1].content if messages else ""

        knowledge = _extract_section(final_message, "Campus knowledge:")
        navigation = _extract_section(final_message, "Navigation status:")
        camera = _extract_section(final_message, "Camera view:")
        dashboard = _extract_section(final_message, "Dashboard:")

        parts: list[str] = []

        if navigation and _NO_NAV_MARKER not in navigation:
            parts.append(navigation)

        if knowledge and _NO_KNOWLEDGE_MARKER not in knowledge:
            parts.append(_first_knowledge_fact(knowledge))

        if dashboard and not any(marker in dashboard for marker in _NO_DASHBOARD_MARKERS):
            parts.append(dashboard)

        if camera and _NO_CAMERA_MARKER not in camera:
            parts.append(camera)

        response_text = " ".join(parts) if parts else _FALLBACK_TEXT

        return MLLMResponse(text=response_text, model="mock-grounded", finish_reason="stop")
