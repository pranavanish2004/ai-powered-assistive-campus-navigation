"""
Multimodal LLM (MLLM) provider abstraction.

Design rationale
-----------------
Per the Phase 1.4 decision: the MLLM layer must be fully pluggable from
day one, selected via Settings, with zero application code depending on
any specific provider's SDK. `MLLMProviderInterface` is that boundary --
`assistant/decision_engine.py` calls this interface only, never
`anthropic`, `openai`, or `google.generativeai` directly.

Just as important, and unchanged since Phase 1.1: whichever provider is
active, it is used **exclusively** for the CONVERSATIONAL lane -- phrasing
a natural-language answer from already-assembled, deterministic context.
It never decides routing, localization, obstacle classification, or any
SAFETY-priority speech. See assistant/decision_engine.py and
assistant/prompt_builder.py for where that boundary is enforced in code,
not just in this docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

MLLMRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class MLLMMessage:
    """One turn in a conversation, in the universal role/content shape
    every major provider's chat API accepts (with minor adaptation inside
    each provider adapter, e.g. Gemini's distinct system-instruction
    handling)."""

    role: MLLMRole
    content: str


@dataclass(frozen=True, slots=True)
class MLLMResponse:
    """A generated response from an MLLM provider."""

    text: str
    model: str
    finish_reason: str | None = None


class MLLMProviderInterface(ABC):
    """Abstract base class for MLLM providers."""

    @abstractmethod
    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        """Generate a response given the full conversation so far
        (typically one system message + retrieved-context/user messages).
        Implementations should raise MLLMError (core.exceptions) on
        failure -- network error, API error, missing credentials, or the
        provider's SDK not being installed -- rather than returning a
        fabricated response."""
