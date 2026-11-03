"""Tests for the MLLM provider factory."""

import pytest

from campus_nav.config.settings import Settings
from campus_nav.infrastructure.mllm.anthropic_provider import AnthropicProvider
from campus_nav.infrastructure.mllm.gemini_provider import GeminiProvider
from campus_nav.infrastructure.mllm.grounded_mock_provider import GroundedMockMLLMProvider
from campus_nav.infrastructure.mllm.local_llama_provider import LocalLlamaProvider
from campus_nav.infrastructure.mllm.mllm_factory import create_mllm_provider
from campus_nav.infrastructure.mllm.openai_provider import OpenAIProvider

pytestmark = pytest.mark.unit


def test_default_provider_is_grounded_mock():
    settings = Settings(_env_file=None)
    assert isinstance(create_mllm_provider(settings), GroundedMockMLLMProvider)


@pytest.mark.parametrize(
    "provider_name,expected_class",
    [
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("gemini", GeminiProvider),
        ("local_llama", LocalLlamaProvider),
        ("mock", GroundedMockMLLMProvider),
    ],
)
def test_factory_selects_correct_provider_class(provider_name, expected_class):
    settings = Settings(_env_file=None, mllm_provider=provider_name)
    assert isinstance(create_mllm_provider(settings), expected_class)


def test_factory_passes_model_and_api_key_to_anthropic():
    settings = Settings(
        _env_file=None,
        mllm_provider="anthropic",
        mllm_model="claude-custom",
        mllm_api_key="sk-test",
    )
    provider = create_mllm_provider(settings)
    assert provider._model == "claude-custom"
    assert provider._api_key == "sk-test"


def test_factory_passes_base_url_to_local_llama():
    settings = Settings(
        _env_file=None, mllm_provider="local_llama", mllm_base_url="http://custom-host:1234"
    )
    provider = create_mllm_provider(settings)
    assert provider._base_url == "http://custom-host:1234"
