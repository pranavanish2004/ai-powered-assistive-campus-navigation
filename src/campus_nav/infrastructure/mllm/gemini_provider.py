"""
Google Gemini MLLM provider adapter.

IMPORTANT -- optional dependency
--------------------------------
Requires the `[llm]` extra. Install with: pip install -e ".[llm]"

Note: Gemini's SDK models system instructions and conversation turns
differently from the Anthropic/OpenAI Messages-list convention (it takes
a distinct `system_instruction` at model-construction time and its own
`role` vocabulary for turns: "user"/"model" rather than "user"/
"assistant"). This adapter translates our universal MLLMMessage list into
that shape internally, so the Decision Engine never needs to know Gemini
is different from any other provider.
"""

from __future__ import annotations

from typing import Any

from campus_nav.core.exceptions import MLLMError, ModelLoadError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMProviderInterface, MLLMResponse

logger = get_logger(__name__)

_INSTALL_HINT = (
    "google-generativeai is not installed. Install the optional llm extra with: "
    'pip install -e ".[llm]"'
)
_DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiProvider(MLLMProviderInterface):
    """MLLM provider backed by Google's Gemini API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model_name = model or _DEFAULT_MODEL
        self._api_key = api_key
        self._genai: Any | None = None

    def _get_genai(self) -> Any:
        if self._genai is None:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                if self._api_key:
                    genai.configure(api_key=self._api_key)
                # else: SDK falls back to GOOGLE_API_KEY.
                self._genai = genai
            except Exception as exc:
                raise ModelLoadError(f"Failed to initialize Gemini client: {exc}") from exc
        return self._genai

    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        genai = self._get_genai()

        system_text = "\n".join(m.content for m in messages if m.role == "system") or None
        history = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [m.content]}
            for m in messages
            if m.role != "system"
        ]

        try:
            model = genai.GenerativeModel(self._model_name, system_instruction=system_text)
            response = model.generate_content(
                history, generation_config={"max_output_tokens": max_tokens}
            )
        except Exception as exc:
            raise MLLMError(f"Gemini API call failed: {exc}") from exc

        return MLLMResponse(text=response.text, model=self._model_name, finish_reason=None)
