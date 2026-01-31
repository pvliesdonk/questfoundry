"""Tests for provider factory."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from questfoundry.providers.base import ProviderError
from questfoundry.providers.factory import (
    PROVIDER_DEFAULTS,
    create_chat_model,
    create_model_for_structured_output,
    get_default_model,
    unload_ollama_model,
)
from questfoundry.providers.model_info import (
    DEFAULT_CONTEXT_WINDOW,
    ModelInfo,
    get_model_info,
)
from questfoundry.providers.structured_output import StructuredOutputStrategy

# --- Tests for get_default_model ---


def test_get_default_model_openai() -> None:
    """OpenAI has a default model."""
    assert get_default_model("openai") == "gpt-5-mini"
    assert get_default_model("OpenAI") == "gpt-5-mini"  # Case insensitive


def test_get_default_model_anthropic() -> None:
    """Anthropic has a default model."""
    assert get_default_model("anthropic") == "claude-sonnet-4-20250514"


def test_get_default_model_ollama_returns_none() -> None:
    """Ollama requires explicit model - returns None."""
    assert get_default_model("ollama") is None


def test_get_default_model_unknown_provider() -> None:
    """Unknown providers return None."""
    assert get_default_model("unknown") is None


def test_provider_defaults_dict_structure() -> None:
    """Provider defaults dict has expected structure."""
    assert "ollama" in PROVIDER_DEFAULTS
    assert "openai" in PROVIDER_DEFAULTS
    assert "anthropic" in PROVIDER_DEFAULTS
    # Ollama should require explicit model
    assert PROVIDER_DEFAULTS["ollama"] is None


# --- Tests for create_chat_model ---


def test_create_chat_model_unknown_provider() -> None:
    """Factory raises error for unknown provider."""
    with pytest.raises(ProviderError) as exc_info:
        create_chat_model("unknown", "model")

    assert "Unknown provider" in str(exc_info.value)
    assert exc_info.value.provider == "unknown"


def test_create_chat_model_ollama_missing_host() -> None:
    """Factory raises error when OLLAMA_HOST not set."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_chat_model("ollama", "qwen3:4b-instruct-32k")

    assert "OLLAMA_HOST not configured" in str(exc_info.value)
    assert exc_info.value.provider == "ollama"


def test_create_chat_model_ollama_success() -> None:
    """Factory creates Ollama chat model."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        result = create_chat_model("ollama", "qwen3:4b-instruct-32k", temperature=0.7)

    assert result is mock_chat
    mock_class.assert_called_once_with(
        model="qwen3:4b-instruct-32k",
        base_url="http://test:11434",
        temperature=0.7,
        num_ctx=32768,
    )


def test_create_chat_model_ollama_with_custom_host() -> None:
    """Factory uses custom host parameter."""
    mock_chat = MagicMock()

    with patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class:
        create_chat_model("ollama", "llama3:8b", host="http://custom:8080", temperature=0.5)

    mock_class.assert_called_once_with(
        model="llama3:8b",
        base_url="http://custom:8080",
        temperature=0.5,
        num_ctx=32768,
    )


def test_create_chat_model_ollama_import_error() -> None:
    """Factory raises error when langchain-ollama not installed."""
    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch(
            "questfoundry.providers.factory._create_ollama_base_model",
            side_effect=ProviderError(
                "ollama",
                "langchain-ollama not installed. Run: uv add langchain-ollama",
            ),
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        create_chat_model("ollama", "model")

    assert "langchain-ollama not installed" in str(exc_info.value)


def test_create_chat_model_openai_missing_key() -> None:
    """Factory raises error when OPENAI_API_KEY not set."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_chat_model("openai", "gpt-5-mini")

    assert "API key required" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


def test_create_chat_model_openai_success() -> None:
    """Factory creates OpenAI chat model."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class,
    ):
        result = create_chat_model("openai", "gpt-5-mini", temperature=0.7)

    assert result is mock_chat
    mock_class.assert_called_once_with(
        model="gpt-5-mini",
        api_key="sk-test",
        temperature=0.7,
    )


def test_create_chat_model_openai_with_custom_key() -> None:
    """Factory uses custom API key parameter."""
    mock_chat = MagicMock()

    with patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class:
        create_chat_model("openai", "gpt-5-mini", api_key="sk-custom")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["api_key"] == "sk-custom"


def test_create_chat_model_openai_import_error() -> None:
    """Factory raises error when langchain-openai not installed."""
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch(
            "questfoundry.providers.factory._create_openai_base_model",
            side_effect=ProviderError(
                "openai",
                "langchain-openai not installed. Run: uv add langchain-openai",
            ),
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        create_chat_model("openai", "model")

    assert "langchain-openai not installed" in str(exc_info.value)


def test_create_chat_model_anthropic_missing_key() -> None:
    """Factory raises error when ANTHROPIC_API_KEY not set."""
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ProviderError) as exc_info:
        create_chat_model("anthropic", "claude-3-opus")

    assert "API key required" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


def test_create_chat_model_anthropic_success() -> None:
    """Factory creates Anthropic chat model."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        patch("langchain_anthropic.ChatAnthropic", return_value=mock_chat) as mock_class,
    ):
        result = create_chat_model("anthropic", "claude-3-opus", temperature=0.7)

    assert result is mock_chat
    mock_class.assert_called_once_with(
        model="claude-3-opus",
        api_key="sk-ant-test",
        temperature=0.7,
    )


def test_create_chat_model_anthropic_import_error() -> None:
    """Factory raises error when langchain-anthropic not installed."""
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        patch(
            "questfoundry.providers.factory._create_anthropic_base_model",
            side_effect=ProviderError(
                "anthropic",
                "langchain-anthropic not installed. Run: uv add langchain-anthropic",
            ),
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        create_chat_model("anthropic", "model")

    assert "langchain-anthropic not installed" in str(exc_info.value)


def test_create_chat_model_case_insensitive() -> None:
    """Factory handles uppercase provider names."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat),
    ):
        result = create_chat_model("OLLAMA", "model", temperature=0.5)

    assert result is not None


def test_create_chat_model_custom_temperature() -> None:
    """Factory passes custom temperature."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("ollama", "model", temperature=0.5)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["temperature"] == 0.5


def test_create_chat_model_ollama_custom_num_ctx() -> None:
    """Factory passes custom num_ctx for context window."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("ollama", "model", num_ctx=131072, temperature=0.5)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["num_ctx"] == 131072


def test_create_chat_model_ollama_no_temperature_when_not_provided() -> None:
    """Factory does not include temperature when not provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("ollama", "model")

    call_kwargs = mock_class.call_args[1]
    assert "temperature" not in call_kwargs


def test_create_chat_model_ollama_top_p() -> None:
    """Factory passes top_p parameter for Ollama."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("ollama", "model", temperature=0.5, top_p=0.95)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["top_p"] == 0.95


def test_create_chat_model_ollama_seed() -> None:
    """Factory passes seed parameter for Ollama."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("ollama", "model", temperature=0.5, seed=42)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["seed"] == 42


def test_create_chat_model_openai_top_p_and_seed() -> None:
    """Factory passes top_p and seed parameters for OpenAI."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("openai", "gpt-5-mini", temperature=0.7, top_p=0.9, seed=123)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["seed"] == 123


def test_create_chat_model_anthropic_top_p() -> None:
    """Factory passes top_p parameter for Anthropic."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        patch("langchain_anthropic.ChatAnthropic", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("anthropic", "claude-3-opus", temperature=0.5, top_p=0.85)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["top_p"] == 0.85


# --- Tests for create_model_for_structured_output ---


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
            model_name="qwen3:4b-instruct-32k",
            schema=SampleSchema,
        )

    assert result is mock_structured
    mock_chat.with_structured_output.assert_called_once()
    call_args = mock_chat.with_structured_output.call_args
    assert call_args[0][0] is SampleSchema
    assert (
        call_args[1]["method"] == "json_schema"
    )  # JSON_MODE for Ollama (better for complex schemas)


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
            model_name="gpt-5-mini",
            schema=SampleSchema,
        )

    assert result is mock_structured
    mock_chat.with_structured_output.assert_called_once()
    call_args = mock_chat.with_structured_output.call_args
    assert (
        call_args[1]["method"] == "function_calling"
    )  # function_calling for OpenAI (handles optional fields)


def test_create_model_structured_without_schema() -> None:
    """Factory returns base model when no schema provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "http://test:11434"}),
        patch("langchain_ollama.ChatOllama", return_value=mock_chat),
    ):
        result = create_model_for_structured_output(
            "ollama",
            model_name="qwen3:4b-instruct-32k",
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
            model_name="gpt-5-mini",
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
    assert call_kwargs["model"] == "qwen3:4b-instruct-32k"


def test_create_model_structured_default_model_openai() -> None:
    """Factory uses default model name for OpenAI when not provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class,
    ):
        create_model_for_structured_output("openai")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "gpt-5-mini"  # Uses PROVIDER_DEFAULTS


def test_create_model_structured_default_model_anthropic() -> None:
    """Factory uses default model name for Anthropic when not provided."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        patch("langchain_anthropic.ChatAnthropic", return_value=mock_chat) as mock_class,
    ):
        create_model_for_structured_output("anthropic")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"  # Uses PROVIDER_DEFAULTS


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


# --- Tests for get_model_info ---


def test_get_model_info_known_model() -> None:
    """get_model_info returns known context window for known models."""
    info = get_model_info("openai", "gpt-5-mini")

    assert info.context_window == 1_000_000
    assert info.supports_tools is True
    assert info.supports_vision is True


def test_get_model_info_unknown_model() -> None:
    """get_model_info returns default for unknown models."""
    info = get_model_info("openai", "unknown-model-xyz")

    assert info.context_window == DEFAULT_CONTEXT_WINDOW
    assert info.supports_tools is True


def test_get_model_info_ollama() -> None:
    """get_model_info works for Ollama models."""
    info = get_model_info("ollama", "qwen3:4b-instruct-32k")

    assert info.context_window == 32_768


def test_get_model_info_anthropic() -> None:
    """get_model_info works for Anthropic models."""
    info = get_model_info("anthropic", "claude-sonnet-4-20250514")

    assert info.context_window == 200_000
    assert info.supports_vision is True


def test_get_model_info_case_insensitive_provider() -> None:
    """get_model_info is case insensitive for provider name."""
    info = get_model_info("OPENAI", "gpt-5-mini")

    assert info.context_window == 1_000_000


def test_model_info_is_frozen() -> None:
    """ModelInfo is immutable."""
    info = ModelInfo(context_window=1000)

    with pytest.raises(FrozenInstanceError):
        info.context_window = 2000  # type: ignore[misc]


# --- Tests for o1 / reasoning model support ---


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        # o1 family
        ("o1", True),
        ("o1-mini", True),
        ("o1-preview", True),
        ("O1", True),
        ("O1-MINI", True),
        # o3 family
        ("o3", True),
        ("o3-mini", True),
        # Non-reasoning models
        ("gpt-5-mini", False),
        ("gpt-4o-mini", False),
        ("gpt-4-turbo", False),
        ("gpt-3.5-turbo", False),
    ],
)
def test_is_reasoning_model(model_name: str, expected: bool) -> None:
    """_is_reasoning_model correctly identifies reasoning models."""
    from questfoundry.providers.factory import _is_reasoning_model

    assert _is_reasoning_model(model_name) is expected


@pytest.mark.parametrize("model_name", ["o1", "o1-mini", "o1-preview", "o3", "o3-mini"])
def test_create_chat_model_reasoning_model_no_temperature(model_name: str) -> None:
    """Factory creates reasoning models without temperature parameter."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("openai", model_name)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == model_name
    assert "temperature" not in call_kwargs


def test_create_chat_model_gpt4o_has_temperature() -> None:
    """Factory creates GPT-4o model with temperature parameter (control case)."""
    mock_chat = MagicMock()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        patch("langchain_openai.ChatOpenAI", return_value=mock_chat) as mock_class,
    ):
        create_chat_model("openai", "gpt-5-mini", temperature=0.7)

    call_kwargs = mock_class.call_args[1]
    # GPT-4o should have temperature when provided
    assert "temperature" in call_kwargs
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.parametrize(
    ("model_name", "expected_context", "expected_tools", "expected_vision"),
    [
        ("o1", 200_000, False, False),
        ("o1-mini", 128_000, False, False),
        ("o1-preview", 128_000, False, False),
        ("o3", 200_000, False, False),
        ("o3-mini", 200_000, False, False),
    ],
)
def test_get_model_info_reasoning_models_no_tools(
    model_name: str, expected_context: int, expected_tools: bool, expected_vision: bool
) -> None:
    """get_model_info returns correct properties for reasoning models."""
    info = get_model_info("openai", model_name)

    assert info.context_window == expected_context
    assert info.supports_tools is expected_tools
    assert info.supports_vision is expected_vision


# --- Tests for unload_ollama_model ---


class TestUnloadOllamaModel:
    """Tests for the Ollama model unloading utility."""

    @pytest.mark.asyncio
    async def test_sends_keep_alive_zero(self) -> None:
        """Sends POST with keep_alive=0 to Ollama API."""
        from unittest.mock import AsyncMock

        mock_model = MagicMock()
        mock_model.base_url = "http://localhost:11434"
        mock_model.model = "qwen3:4b"

        mock_response = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            await unload_ollama_model(mock_model)

            mock_client.post.assert_called_once_with(
                "http://localhost:11434/api/generate",
                json={"model": "qwen3:4b", "keep_alive": 0},
            )

    @pytest.mark.asyncio
    async def test_noop_for_non_ollama_model(self) -> None:
        """Silently returns when model has no base_url (e.g., OpenAI)."""
        mock_model = MagicMock(spec=[])  # no attributes

        with patch("httpx.AsyncClient") as mock_cls:
            await unload_ollama_model(mock_model)
            mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_base_url_is_none(self) -> None:
        """Returns when base_url is None."""
        mock_model = MagicMock()
        mock_model.base_url = None
        mock_model.model = "qwen3:4b"

        with patch("httpx.AsyncClient") as mock_cls:
            await unload_ollama_model(mock_model)
            mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_warning_on_failure(self) -> None:
        """Logs warning but doesn't raise on connection failure."""
        from unittest.mock import AsyncMock

        mock_model = MagicMock()
        mock_model.base_url = "http://localhost:11434"
        mock_model.model = "qwen3:4b"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            # Should not raise
            await unload_ollama_model(mock_model)
