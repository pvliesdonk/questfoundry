"""Tests for provider factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from questfoundry.providers.base import ProviderError
from questfoundry.providers.factory import create_model_for_structured_output, create_provider
from questfoundry.providers.structured_output import StructuredOutputStrategy


def test_create_unknown_provider() -> None:
    """Factory raises error for unknown provider."""
    with pytest.raises(ProviderError) as exc_info:
        create_provider("unknown", "model")

    assert "Unknown provider" in str(exc_info.value)
    assert exc_info.value.provider == "unknown"


def test_create_ollama_missing_host() -> None:
    """Factory raises error when OLLAMA_HOST not set."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_provider("ollama", "qwen3:8b")

    assert "OLLAMA_HOST not configured" in str(exc_info.value)
    assert exc_info.value.provider == "ollama"


def test_create_ollama_success() -> None:
    """Factory creates Ollama provider."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama") as mock_chat,
    ):
        mock_chat.return_value = MagicMock()
        provider = create_provider("ollama", "qwen3:8b")

    assert provider.default_model == "qwen3:8b"
    mock_chat.assert_called_once_with(
        model="qwen3:8b",
        base_url="http://test:11434",
        temperature=0.7,
    )


def test_create_ollama_with_custom_host() -> None:
    """Factory uses custom host parameter."""
    with patch("langchain_ollama.ChatOllama") as mock_chat:
        mock_chat.return_value = MagicMock()
        provider = create_provider("ollama", "llama3:8b", host="http://custom:8080")

    assert provider.default_model == "llama3:8b"
    mock_chat.assert_called_once_with(
        model="llama3:8b",
        base_url="http://custom:8080",
        temperature=0.7,
    )


def test_create_ollama_import_error() -> None:
    """Factory raises error when langchain-ollama not installed."""
    import questfoundry.providers.factory as factory_module

    original = factory_module._create_ollama

    def mock_create_ollama(_model: str, **_kwargs: object) -> None:
        raise ProviderError(
            "ollama",
            "langchain-ollama not installed. Run: uv add langchain-ollama",
        )

    factory_module._create_ollama = mock_create_ollama  # type: ignore[assignment]
    try:
        with pytest.raises(ProviderError) as exc_info:
            create_provider("ollama", "model", host="http://test:11434")

        assert "langchain-ollama not installed" in str(exc_info.value)
    finally:
        factory_module._create_ollama = original


def test_create_openai_missing_key() -> None:
    """Factory raises error when OPENAI_API_KEY not set."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_provider("openai", "gpt-4o")

    assert "API key required" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


def test_create_openai_success() -> None:
    """Factory creates OpenAI provider."""
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI") as mock_chat,
    ):
        mock_chat.return_value = MagicMock()
        provider = create_provider("openai", "gpt-4o")

    assert provider.default_model == "gpt-4o"
    mock_chat.assert_called_once_with(
        model="gpt-4o",
        api_key="sk-test",
        temperature=0.7,
    )


def test_create_openai_with_custom_key() -> None:
    """Factory uses custom API key parameter."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_chat.return_value = MagicMock()
        provider = create_provider("openai", "gpt-4o-mini", api_key="sk-custom")

    assert provider.default_model == "gpt-4o-mini"
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["api_key"] == "sk-custom"


def test_create_openai_import_error() -> None:
    """Factory raises error when langchain-openai not installed."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        import questfoundry.providers.factory as factory_module

        original_create_openai = factory_module._create_openai

        def mock_create_openai(_model: str, **_kwargs: object) -> None:
            raise ProviderError(
                "openai",
                "langchain-openai not installed. Run: uv add langchain-openai",
            )

        factory_module._create_openai = mock_create_openai
        try:
            with pytest.raises(ProviderError) as exc_info:
                create_provider("openai", "model")

            assert "langchain-openai not installed" in str(exc_info.value)
        finally:
            factory_module._create_openai = original_create_openai


def test_create_anthropic_missing_key() -> None:
    """Factory raises error when ANTHROPIC_API_KEY not set."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_provider("anthropic", "claude-3-opus")

    assert "API key required" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


def test_create_anthropic_success() -> None:
    """Factory creates Anthropic provider."""
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        patch("langchain_anthropic.ChatAnthropic") as mock_chat,
    ):
        mock_chat.return_value = MagicMock()
        provider = create_provider("anthropic", "claude-3-opus")

    assert provider.default_model == "claude-3-opus"
    mock_chat.assert_called_once_with(
        model="claude-3-opus",
        api_key="sk-ant-test",
        temperature=0.7,
    )


def test_create_anthropic_import_error() -> None:
    """Factory raises error when langchain-anthropic not installed."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        import questfoundry.providers.factory as factory_module

        original_create_anthropic = factory_module._create_anthropic

        def mock_create_anthropic(_model: str, **_kwargs: object) -> None:
            raise ProviderError(
                "anthropic",
                "langchain-anthropic not installed. Run: uv add langchain-anthropic",
            )

        factory_module._create_anthropic = mock_create_anthropic
        try:
            with pytest.raises(ProviderError) as exc_info:
                create_provider("anthropic", "model")

            assert "langchain-anthropic not installed" in str(exc_info.value)
        finally:
            factory_module._create_anthropic = original_create_anthropic


def test_provider_name_case_insensitive() -> None:
    """Factory handles uppercase provider names."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama") as mock_chat,
    ):
        mock_chat.return_value = MagicMock()
        provider = create_provider("OLLAMA", "model")

    assert provider is not None


def test_provider_custom_temperature() -> None:
    """Factory passes custom temperature."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama") as mock_chat,
    ):
        mock_chat.return_value = MagicMock()
        create_provider("ollama", "model", temperature=0.5)

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["temperature"] == 0.5


# Tests for create_model_for_structured_output


class SampleSchema(BaseModel):
    """Sample schema for structured output tests."""

    title: str
    count: int


def test_create_model_structured_ollama_with_schema() -> None:
    """Factory creates Ollama model with structured output."""
    mock_chat = MagicMock()
    mock_structured = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat),
    ):
        result = create_model_for_structured_output(
            "ollama",
            model_name="qwen3:8b",
            schema=SampleSchema,
        )

    assert result is mock_structured
    mock_chat.with_structured_output.assert_called_once()
    call_args = mock_chat.with_structured_output.call_args
    assert call_args[0][0] is SampleSchema
    assert call_args[1]["method"] == "function_calling"  # Tool strategy for Ollama


def test_create_model_structured_openai_with_schema() -> None:
    """Factory creates OpenAI model with structured output."""
    mock_chat = MagicMock()
    mock_structured = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat),
    ):
        result = create_model_for_structured_output(
            "openai",
            model_name="gpt-4o",
            schema=SampleSchema,
        )

    assert result is mock_structured
    mock_chat.with_structured_output.assert_called_once()
    call_args = mock_chat.with_structured_output.call_args
    assert call_args[1]["method"] == "json_mode"  # JSON mode for OpenAI


def test_create_model_structured_without_schema() -> None:
    """Factory returns base model when no schema provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat),
    ):
        result = create_model_for_structured_output(
            "ollama",
            model_name="qwen3:8b",
            schema=None,
        )

    assert result is mock_chat
    mock_chat.with_structured_output.assert_not_called()


def test_create_model_structured_explicit_strategy() -> None:
    """Factory uses explicit strategy over auto-detect."""
    mock_chat = MagicMock()
    mock_structured = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat),
    ):
        # Force tool strategy on OpenAI (normally would use JSON mode)
        result = create_model_for_structured_output(
            "openai",
            model_name="gpt-4o",
            schema=SampleSchema,
            strategy=StructuredOutputStrategy.TOOL,
        )

    assert result is mock_structured
    call_args = mock_chat.with_structured_output.call_args
    assert call_args[1]["method"] == "function_calling"


def test_create_model_structured_default_model_ollama() -> None:
    """Factory uses default model name for Ollama when not provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        create_model_for_structured_output("ollama")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "qwen3:8b"


def test_create_model_structured_default_model_openai() -> None:
    """Factory uses default model name for OpenAI when not provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class,
    ):
        create_model_for_structured_output("openai")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"


def test_create_model_structured_default_model_anthropic() -> None:
    """Factory uses default model name for Anthropic when not provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        patch("langchain_anthropic.ChatAnthropic", return_value=mock_chat) as mock_class,
    ):
        create_model_for_structured_output("anthropic")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "claude-3-5-sonnet"


def test_create_model_structured_unknown_provider() -> None:
    """Factory raises error for unknown provider."""
    with pytest.raises(ProviderError) as exc_info:
        create_model_for_structured_output("unknown", model_name="model")

    assert "Unknown provider" in str(exc_info.value)


def test_create_model_structured_missing_config() -> None:
    """Factory raises error when provider config is missing."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_model_for_structured_output("ollama", model_name="model")

    assert "OLLAMA_HOST not configured" in str(exc_info.value)
