"""MLLM provider factory -- Settings-driven selection. This is the ONE
place in the codebase that knows which provider is active; every other
module (the Decision Engine, the API layer) depends only on
MLLMProviderInterface."""

from __future__ import annotations

from campus_nav.config.settings import Settings
from campus_nav.core.exceptions import ConfigurationError
from campus_nav.infrastructure.mllm.anthropic_provider import AnthropicProvider
from campus_nav.infrastructure.mllm.gemini_provider import GeminiProvider
from campus_nav.infrastructure.mllm.grounded_mock_provider import GroundedMockMLLMProvider
from campus_nav.infrastructure.mllm.local_llama_provider import LocalLlamaProvider
from campus_nav.infrastructure.mllm.openai_provider import OpenAIProvider
from campus_nav.interfaces.mllm_interface import MLLMProviderInterface


def create_mllm_provider(settings: Settings) -> MLLMProviderInterface:
    if settings.mllm_provider == "anthropic":
        return AnthropicProvider(model=settings.mllm_model, api_key=settings.mllm_api_key)
    if settings.mllm_provider == "openai":
        return OpenAIProvider(model=settings.mllm_model, api_key=settings.mllm_api_key)
    if settings.mllm_provider == "gemini":
        return GeminiProvider(model=settings.mllm_model, api_key=settings.mllm_api_key)
    if settings.mllm_provider == "local_llama":
        return LocalLlamaProvider(model=settings.mllm_model, base_url=settings.mllm_base_url)
    if settings.mllm_provider == "mock":
        return GroundedMockMLLMProvider()

    # Unreachable given Settings' Literal type validates this at parse
    # time, but kept explicit rather than falling through silently.
    raise ConfigurationError(f"Unknown mllm_provider: {settings.mllm_provider!r}")
