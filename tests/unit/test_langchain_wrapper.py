"""Tests for LangChain wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.providers.base import Message, ProviderError
from questfoundry.providers.langchain_wrapper import LangChainProvider


@pytest.fixture
def mock_ai_message() -> MagicMock:
    """Create mock AIMessage response."""
    msg = MagicMock()
    msg.content = "Hello, world!"
    msg.usage_metadata = {"total_tokens": 100}
    msg.response_metadata = {"finish_reason": "stop"}
    return msg


@pytest.fixture
def mock_chat_model(mock_ai_message: MagicMock) -> MagicMock:
    """Create mock LangChain chat model."""
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=mock_ai_message)
    return model


@pytest.mark.asyncio
async def test_complete_success(mock_chat_model: MagicMock) -> None:
    """LangChainProvider successfully completes."""
    provider = LangChainProvider(mock_chat_model, "test-model")

    messages: list[Message] = [{"role": "user", "content": "Hi"}]
    result = await provider.complete(messages)

    assert result.content == "Hello, world!"
    assert result.tokens_used == 100
    assert result.finish_reason == "stop"
    assert result.model == "test-model"
    assert result.is_complete


@pytest.mark.asyncio
async def test_complete_with_system_message(mock_chat_model: MagicMock) -> None:
    """LangChainProvider converts system messages correctly."""
    provider = LangChainProvider(mock_chat_model, "test-model")

    messages: list[Message] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    await provider.complete(messages)

    call_args = mock_chat_model.ainvoke.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].__class__.__name__ == "SystemMessage"
    assert call_args[1].__class__.__name__ == "HumanMessage"


@pytest.mark.asyncio
async def test_complete_with_assistant_message(mock_chat_model: MagicMock) -> None:
    """LangChainProvider converts assistant messages correctly."""
    provider = LangChainProvider(mock_chat_model, "test-model")

    messages: list[Message] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    await provider.complete(messages)

    call_args = mock_chat_model.ainvoke.call_args[0][0]
    assert len(call_args) == 3
    assert call_args[0].__class__.__name__ == "HumanMessage"
    assert call_args[1].__class__.__name__ == "AIMessage"
    assert call_args[2].__class__.__name__ == "HumanMessage"


@pytest.mark.asyncio
async def test_complete_no_usage_metadata(mock_chat_model: MagicMock) -> None:
    """LangChainProvider handles missing usage metadata."""
    mock_chat_model.ainvoke.return_value.usage_metadata = None
    provider = LangChainProvider(mock_chat_model, "test-model")

    result = await provider.complete([{"role": "user", "content": "Hi"}])

    assert result.tokens_used == 0


@pytest.mark.asyncio
async def test_complete_no_response_metadata(mock_chat_model: MagicMock) -> None:
    """LangChainProvider handles missing response metadata."""
    mock_chat_model.ainvoke.return_value.response_metadata = None
    provider = LangChainProvider(mock_chat_model, "test-model")

    result = await provider.complete([{"role": "user", "content": "Hi"}])

    assert result.finish_reason == "stop"  # Default


@pytest.mark.asyncio
async def test_complete_list_content(mock_chat_model: MagicMock) -> None:
    """LangChainProvider handles list content (multi-modal responses)."""
    mock_chat_model.ainvoke.return_value.content = ["Hello", " ", "world!"]
    provider = LangChainProvider(mock_chat_model, "test-model")

    result = await provider.complete([{"role": "user", "content": "Hi"}])

    assert result.content == "Hello world!"


@pytest.mark.asyncio
async def test_complete_error_handling(mock_chat_model: MagicMock) -> None:
    """LangChainProvider wraps errors in ProviderError."""
    mock_chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
    provider = LangChainProvider(mock_chat_model, "test-model")

    with pytest.raises(ProviderError) as exc_info:
        await provider.complete([{"role": "user", "content": "Hi"}])

    assert "Completion failed" in str(exc_info.value)
    assert exc_info.value.provider == "langchain"


@pytest.mark.asyncio
async def test_complete_with_model_override(mock_chat_model: MagicMock) -> None:
    """LangChainProvider records model override in response."""
    provider = LangChainProvider(mock_chat_model, "default-model")

    result = await provider.complete(
        [{"role": "user", "content": "Hi"}],
        model="override-model",
    )

    # The model override is recorded in the response
    assert result.model == "override-model"


@pytest.mark.asyncio
async def test_context_manager() -> None:
    """LangChainProvider works as context manager."""
    model = MagicMock()
    async with LangChainProvider(model, "test") as provider:
        assert provider.default_model == "test"


def test_default_model_property() -> None:
    """LangChainProvider exposes default_model."""
    model = MagicMock()
    provider = LangChainProvider(model, "my-model")
    assert provider.default_model == "my-model"


def test_to_langchain_message_unknown_role() -> None:
    """LangChainProvider raises on unknown role."""
    model = MagicMock()
    provider = LangChainProvider(model, "test")

    with pytest.raises(ValueError, match="Unknown role"):
        provider._to_langchain_message({"role": "unknown", "content": "test"})  # type: ignore[typeddict-item]
