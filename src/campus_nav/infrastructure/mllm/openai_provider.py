"""
OpenAI MLLM provider adapter.

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
    'openai is not installed. Install the optional llm extra with: pip install -e ".[llm]"'
)
_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(MLLMProviderInterface):
    """MLLM provider backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or _DEFAULT_MODEL
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                # api_key=None lets the SDK fall back to OPENAI_API_KEY.
                self._client = openai.OpenAI(api_key=self._api_key)
            except Exception as exc:
                raise ModelLoadError(f"Failed to initialize OpenAI client: {exc}") from exc
        return self._client

    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        client = self._get_client()
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=chat_messages,
            )
        except Exception as exc:
            raise MLLMError(f"OpenAI API call failed: {exc}") from exc

        choice = response.choices[0]
        return MLLMResponse(
            text=choice.message.content or "",
            model=self._model,
            finish_reason=choice.finish_reason,
        )
