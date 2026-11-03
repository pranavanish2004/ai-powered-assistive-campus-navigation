"""Tests for the real MLLM provider adapters.

Two things are tested per SDK-backed provider (Anthropic/OpenAI/Gemini):
1. The "SDK not installed" path -- true out of the box in this test
   environment, since the optional [llm] extra is not part of the default
   install, exactly like WhisperSTT/YOLOv8Detector in earlier phases.
2. The "successful generation" path, by injecting a fake SDK module into
   sys.modules -- avoids needing real API credentials or network access,
   and avoids installing the real (if lightweight) SDKs just to test our
   own adapter code around them.

LocalLlamaProvider needs neither: it uses the already-core httpx
dependency directly, tested by mocking httpx.post (same pattern as
IITHDashboardRepository's tests).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from campus_nav.core.exceptions import MLLMError, ModelLoadError
from campus_nav.infrastructure.mllm.anthropic_provider import AnthropicProvider
from campus_nav.infrastructure.mllm.gemini_provider import GeminiProvider
from campus_nav.infrastructure.mllm.local_llama_provider import LocalLlamaProvider
from campus_nav.infrastructure.mllm.openai_provider import OpenAIProvider
from campus_nav.interfaces.mllm_interface import MLLMMessage

pytestmark = pytest.mark.unit


# --- AnthropicProvider ---


def test_anthropic_provider_raises_model_load_error_when_sdk_not_installed():
    with patch.dict(sys.modules, {"anthropic": None}):
        provider = AnthropicProvider()
        with pytest.raises(ModelLoadError):
            provider.generate([MLLMMessage(role="user", content="Hi")])


def test_anthropic_provider_generates_successfully_with_mocked_sdk():
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello from Claude.")],
        stop_reason="end_turn",
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic_module = SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))

    with patch.dict(sys.modules, {"anthropic": fake_anthropic_module}):
        provider = AnthropicProvider(model="claude-test")
        response = provider.generate(
            [
                MLLMMessage(role="system", content="Be helpful."),
                MLLMMessage(role="user", content="Hi"),
            ]
        )

    assert response.text == "Hello from Claude."
    assert response.model == "claude-test"
    assert response.finish_reason == "end_turn"


def test_anthropic_provider_wraps_api_failure_as_mllm_error():
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("simulated API failure")
    fake_anthropic_module = SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))

    with patch.dict(sys.modules, {"anthropic": fake_anthropic_module}):
        provider = AnthropicProvider()
        with pytest.raises(MLLMError):
            provider.generate([MLLMMessage(role="user", content="Hi")])


# --- OpenAIProvider ---


def test_openai_provider_raises_model_load_error_when_sdk_not_installed():
    with patch.dict(sys.modules, {"openai": None}):
        provider = OpenAIProvider()
        with pytest.raises(ModelLoadError):
            provider.generate([MLLMMessage(role="user", content="Hi")])


def test_openai_provider_generates_successfully_with_mocked_sdk():
    fake_message = SimpleNamespace(content="Hello from GPT.")
    fake_choice = SimpleNamespace(message=fake_message, finish_reason="stop")
    fake_response = SimpleNamespace(choices=[fake_choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    fake_openai_module = SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))

    with patch.dict(sys.modules, {"openai": fake_openai_module}):
        provider = OpenAIProvider(model="gpt-test")
        response = provider.generate([MLLMMessage(role="user", content="Hi")])

    assert response.text == "Hello from GPT."
    assert response.finish_reason == "stop"


def test_openai_provider_wraps_api_failure_as_mllm_error():
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("simulated failure")
    fake_openai_module = SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))

    with patch.dict(sys.modules, {"openai": fake_openai_module}):
        provider = OpenAIProvider()
        with pytest.raises(MLLMError):
            provider.generate([MLLMMessage(role="user", content="Hi")])


# --- GeminiProvider ---


def test_gemini_provider_raises_model_load_error_when_sdk_not_installed():
    with patch.dict(sys.modules, {"google.generativeai": None, "google": None}):
        provider = GeminiProvider()
        with pytest.raises(ModelLoadError):
            provider.generate([MLLMMessage(role="user", content="Hi")])


def test_gemini_provider_generates_successfully_with_mocked_sdk():
    fake_model_instance = MagicMock()
    fake_model_instance.generate_content.return_value = SimpleNamespace(text="Hello from Gemini.")

    fake_genai_module = SimpleNamespace(
        configure=MagicMock(),
        GenerativeModel=MagicMock(return_value=fake_model_instance),
    )
    fake_google_module = SimpleNamespace(generativeai=fake_genai_module)

    with patch.dict(
        sys.modules, {"google": fake_google_module, "google.generativeai": fake_genai_module}
    ):
        provider = GeminiProvider(model="gemini-test", api_key="fake-key")
        response = provider.generate([MLLMMessage(role="user", content="Hi")])

    assert response.text == "Hello from Gemini."
    fake_genai_module.configure.assert_called_once_with(api_key="fake-key")


def test_gemini_provider_wraps_api_failure_as_mllm_error():
    fake_model_instance = MagicMock()
    fake_model_instance.generate_content.side_effect = RuntimeError("simulated failure")
    fake_genai_module = SimpleNamespace(
        configure=MagicMock(), GenerativeModel=MagicMock(return_value=fake_model_instance)
    )

    with patch.dict(
        sys.modules,
        {
            "google": SimpleNamespace(generativeai=fake_genai_module),
            "google.generativeai": fake_genai_module,
        },
    ):
        provider = GeminiProvider()
        with pytest.raises(MLLMError):
            provider.generate([MLLMMessage(role="user", content="Hi")])


# --- LocalLlamaProvider (Ollama over HTTP, no optional SDK) ---


@patch("campus_nav.infrastructure.mllm.local_llama_provider.httpx.post")
def test_local_llama_provider_generates_successfully(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "message": {"content": "Hello from local Llama."},
        "done_reason": "stop",
    }
    mock_post.return_value = mock_response

    provider = LocalLlamaProvider(model="llama3-test", base_url="http://localhost:11434")
    response = provider.generate([MLLMMessage(role="user", content="Hi")])

    assert response.text == "Hello from local Llama."
    assert response.model == "llama3-test"
    assert response.finish_reason == "stop"
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "http://localhost:11434/api/chat"


@patch("campus_nav.infrastructure.mllm.local_llama_provider.httpx.post")
def test_local_llama_provider_connection_failure_raises_mllm_error(mock_post):
    mock_post.side_effect = httpx.ConnectError("connection refused")

    provider = LocalLlamaProvider(base_url="http://localhost:11434")
    with pytest.raises(MLLMError):
        provider.generate([MLLMMessage(role="user", content="Hi")])


@patch("campus_nav.infrastructure.mllm.local_llama_provider.httpx.post")
def test_local_llama_provider_unexpected_response_shape_raises_mllm_error(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response

    provider = LocalLlamaProvider()
    with pytest.raises(MLLMError):
        provider.generate([MLLMMessage(role="user", content="Hi")])
