"""Tests for the MLLM interface contract and MockMLLMProvider."""

import pytest

from campus_nav.infrastructure.mllm.mock_mllm_provider import MockMLLMProvider
from campus_nav.interfaces.mllm_interface import MLLMMessage, MLLMProviderInterface

pytestmark = pytest.mark.unit


def test_cannot_instantiate_mllm_interface_directly():
    with pytest.raises(TypeError):
        MLLMProviderInterface()  # type: ignore[abstract]


def test_mock_provider_returns_fixed_response():
    provider = MockMLLMProvider()
    response = provider.generate([MLLMMessage(role="user", content="Hi")])
    assert response.text == "This is a mock assistant response."
    assert response.model == "mock"


def test_mock_provider_records_last_messages():
    provider = MockMLLMProvider()
    messages = [
        MLLMMessage(role="system", content="You are helpful."),
        MLLMMessage(role="user", content="Where is the library?"),
    ]
    provider.generate(messages)
    assert provider.last_messages == messages


def test_set_response_changes_subsequent_output():
    provider = MockMLLMProvider()
    provider.set_response("Custom response.")
    response = provider.generate([MLLMMessage(role="user", content="Hi")])
    assert response.text == "Custom response."


def test_mllm_message_is_immutable():
    message = MLLMMessage(role="user", content="Hi")
    with pytest.raises(AttributeError):
        message.content = "changed"  # type: ignore[misc]
