"""
Local Llama MLLM provider adapter -- talks to a locally-running Ollama
server over plain HTTP.

Why Ollama, and why no optional extra
---------------------------------------
Running a local Llama model directly (llama-cpp-python) means compiling
platform-specific native bindings, which is exactly the kind of heavy,
fragile dependency this project has avoided everywhere else (see
Phase 1.1's note on why openai-whisper was moved out of the default
install). Ollama runs as a separate local server process and exposes a
plain REST API -- this adapter needs nothing beyond the already-core
`httpx` dependency. The trade-off: Ollama itself must be installed and
running separately; this adapter cannot install or start it for you, and
raises a clear MLLMError (not a confusing connection-refused traceback)
if it isn't reachable.
"""

from __future__ import annotations

import httpx

from campus_nav.core.exceptions import MLLMError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMProviderInterface, MLLMResponse

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3"


class LocalLlamaProvider(MLLMProviderInterface):
    """MLLM provider backed by a local Ollama server's `/api/chat` endpoint."""

    def __init__(
        self, model: str | None = None, base_url: str | None = None, timeout_seconds: float = 30.0
    ) -> None:
        self._model = model or _DEFAULT_MODEL
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds

    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        try:
            response = httpx.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise MLLMError(
                f"Could not reach local Ollama server at {self._base_url!r}: {exc}"
            ) from exc
        except ValueError as exc:
            raise MLLMError(f"Ollama returned a non-JSON response: {exc}") from exc

        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise MLLMError(f"Unexpected Ollama response shape: {exc}") from exc

        return MLLMResponse(text=text, model=self._model, finish_reason=data.get("done_reason"))
