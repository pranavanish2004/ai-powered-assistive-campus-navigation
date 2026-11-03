"""Mock MLLM provider -- deterministic, network-free default."""

from __future__ import annotations

from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMProviderInterface, MLLMResponse


class MockMLLMProvider(MLLMProviderInterface):
    """Returns a fixed or caller-configurable response, ignoring the
    actual conversation content (beyond optionally echoing it back for
    debugging). Safe default requiring no API key or network access."""

    def __init__(self, fixed_response_text: str = "This is a mock assistant response.") -> None:
        self._fixed_response_text = fixed_response_text
        self.last_messages: list[MLLMMessage] = []
        """Records the most recent call's messages -- lets tests assert
        exactly what the Decision Engine assembled and sent, without
        needing a real provider or network access."""

    def set_response(self, text: str) -> None:
        self._fixed_response_text = text

    def generate(self, messages: list[MLLMMessage], max_tokens: int = 512) -> MLLMResponse:
        self.last_messages = list(messages)
        return MLLMResponse(text=self._fixed_response_text, model="mock", finish_reason="stop")
