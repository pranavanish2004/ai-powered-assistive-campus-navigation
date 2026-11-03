"""
Anthropic (Claude) MLLM provider adapter.

IMPORTANT -- optional dependency
--------------------------------
Requires the `[llm]` extra. Install with: pip install -e ".[llm]"
"""

from __future__ import annotations

from typing import Any

from campus_nav.core.exceptions import MLLMError, ModelLoadError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMProviderInterface, MLLMResponse

logger = get_logger(__name__)

_INSTALL_HINT = (
    'anthropic is not installed. Install the optional llm extra with: pip install -e ".[llm]"'
)
_DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicProvider(MLLMProviderInterface):
    """MLLM provider backed by the Anthropic Messages API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or _DEFAULT_MODEL
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                # api_key=None lets the SDK fall back to ANTHROPIC_API_KEY.
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except Exception as exc:
                raise ModelLoadError(f"Failed to initialize Anthropic client: {exc}") from exc
        return self._client

    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        client = self._get_client()

        system_text = "\n".join(m.content for m in messages if m.role == "system")
        chat_messages = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_text or None,
                messages=chat_messages,
            )
        except Exception as exc:
            raise MLLMError(f"Anthropic API call failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return MLLMResponse(text=text, model=self._model, finish_reason=response.stop_reason)
