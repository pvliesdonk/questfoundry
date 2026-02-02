"""Tests for provider factory."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from questfoundry.providers.base import ProviderError
from questfoundry.providers.factory import (
    PROVIDER_DEFAULTS,
    _normalize_provider,
    _query_ollama_num_ctx,
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


def test_get_default_model_google() -> None:
    """Google has a default model."""
    assert get_default_model("google") == "gemini-2.5-flash"


def test_get_default_model_unknown_provider() -> None:
    """Unknown providers return None."""
    assert get_default_model("unknown") is None


def test_provider_defaults_dict_structure() -> None:
    """Provider defaults dict has expected structure."""
    assert "ollama" in PROVIDER_DEFAULTS
    assert "openai" in PROVIDER_DEFAULTS
    assert "anthropic" in PROVIDER_DEFAULTS
    assert "google" in PROVIDER_DEFAULTS
    # Ollama should require explicit model
    assert PROVIDER_DEFAULTS["ollama"] is None


# --- Tests for _normalize_provider (alias resolution) ---


def test_normalize_provider_gemini_alias() -> None:
    """'gemini' resolves to 'google'."""
    assert _normalize_provider("gemini") == "google"
    assert _normalize_provider("Gemini") == "google"
    assert _normalize_provider("GEMINI") == "google"


def test_normalize_provider_passthrough() -> None:
    """Other provider names pass through lowercased."""
    assert _normalize_provider("openai") == "openai"
    assert _normalize_provider("OPENAI") == "openai"
    assert _normalize_provider("google") == "google"
    assert _normalize_provider("ollama") == "ollama"


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


def test_create_chat_model_google_missing_key() -> None:
    """Factory raises error when GOOGLE_API_KEY not set."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "questfoundry.providers.factory._create_google_base_model",
            side_effect=ProviderError(
                "google",
                "API key required. Set GOOGLE_API_KEY environment variable.",
            ),
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        create_chat_model("google", "gemini-2.5-flash")

    assert "API key required" in str(exc_info.value)
    assert exc_info.value.provider == "google"


def _mock_google_module() -> tuple[ModuleType, MagicMock]:
    """Create a mock langchain_google_genai module with ChatGoogleGenerativeAI."""
    mock_module = ModuleType("langchain_google_genai")
    mock_class = MagicMock()
    mock_module.ChatGoogleGenerativeAI = mock_class  # type: ignore[attr-defined]
    return mock_module, mock_class


def test_create_chat_model_google_success() -> None:
    """Factory creates Google Gemini chat model."""
    mock_chat = MagicMock()
    mock_module, mock_class = _mock_google_module()
    mock_class.return_value = mock_chat

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch.dict(sys.modules, {"langchain_google_genai": mock_module}),
    ):
        result = create_chat_model("google", "gemini-2.5-flash", temperature=0.7)

    assert result is mock_chat
    mock_class.assert_called_once_with(
        model="gemini-2.5-flash",
        google_api_key="test-key",
        temperature=0.7,
    )


def test_create_chat_model_google_import_error() -> None:
    """Factory raises error when langchain-google-genai not installed."""
    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch(
            "questfoundry.providers.factory._create_google_base_model",
            side_effect=ProviderError(
                "google",
                "langchain-google-genai not installed. Run: uv add langchain-google-genai",
            ),
        ),
        pytest.raises(ProviderError) as exc_info,
    ):
        create_chat_model("google", "model")

    assert "langchain-google-genai not installed" in str(exc_info.value)


def test_create_chat_model_google_top_p() -> None:
    """Factory passes top_p parameter for Google."""
    mock_chat = MagicMock()
    mock_module, mock_class = _mock_google_module()
    mock_class.return_value = mock_chat

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch.dict(sys.modules, {"langchain_google_genai": mock_module}),
    ):
        create_chat_model("google", "gemini-2.5-flash", temperature=0.5, top_p=0.9)

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["top_p"] == 0.9


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


def test_create_model_structured_google_with_schema() -> None:
    """Factory creates Google model with structured output using JSON_MODE."""
    mock_chat = MagicMock()
    mock_structured = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured
    mock_module, mock_class = _mock_google_module()
    mock_class.return_value = mock_chat

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch.dict(sys.modules, {"langchain_google_genai": mock_module}),
    ):
        result = create_model_for_structured_output(
            "google",
            model_name="gemini-2.5-flash",
            schema=SampleSchema,
        )

    assert result is mock_structured
    call_args = mock_chat.with_structured_output.call_args
    assert call_args[0][0] is SampleSchema
    assert call_args[1]["method"] == "json_schema"  # JSON_MODE for Google


def test_create_model_structured_default_model_google() -> None:
    """Factory uses default model name for Google when not provided."""
    mock_chat = MagicMock()
    mock_module, mock_class = _mock_google_module()
    mock_class.return_value = mock_chat

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch.dict(sys.modules, {"langchain_google_genai": mock_module}),
    ):
        create_model_for_structured_output("google")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "gemini-2.5-flash"  # Uses PROVIDER_DEFAULTS


def test_create_model_structured_gemini_alias() -> None:
    """Factory handles 'gemini' alias in structured output creation."""
    mock_chat = MagicMock()
    mock_module, mock_class = _mock_google_module()
    mock_class.return_value = mock_chat

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch.dict(sys.modules, {"langchain_google_genai": mock_module}),
    ):
        create_model_for_structured_output("gemini")

    call_kwargs = mock_class.call_args[1]
    assert call_kwargs["model"] == "gemini-2.5-flash"


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


def test_get_model_info_google() -> None:
    """get_model_info works for Google Gemini models."""
    info = get_model_info("google", "gemini-2.5-flash")

    assert info.context_window == 1_000_000
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


# --- Tests for _query_ollama_num_ctx ---


class TestQueryOllamaNumCtx:
    """Tests for querying Ollama /api/show for num_ctx."""

    def test_returns_num_ctx_from_parameters(self) -> None:
        """Extracts num_ctx from the parameters field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parameters": "temperature  0.7\nnum_ctx  32768\nrepeat_penalty  1",
            "model_info": {},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=None)

            result = _query_ollama_num_ctx("http://localhost:11434", "qwen3:4b")

        assert result == 32768

    def test_falls_back_to_arch_context_length(self) -> None:
        """Uses model_info context_length when parameters lacks num_ctx."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parameters": "temperature  0.7",
            "model_info": {"qwen3.context_length": 262144},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=None)

            result = _query_ollama_num_ctx("http://localhost:11434", "qwen3:4b")

        assert result == 262144

    def test_returns_none_on_connection_error(self) -> None:
        """Returns None when Ollama is unreachable."""
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(side_effect=ConnectionError("refused"))

            result = _query_ollama_num_ctx("http://localhost:11434", "qwen3:4b")

        assert result is None

    def test_returns_none_when_no_context_info(self) -> None:
        """Returns None when response has no num_ctx or context_length."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parameters": "temperature  0.7",
            "model_info": {"general.architecture": "qwen3"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=None)

            result = _query_ollama_num_ctx("http://localhost:11434", "qwen3:4b")

        assert result is None

    def test_prefers_parameters_over_arch(self) -> None:
        """parameters num_ctx takes precedence over model_info context_length."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parameters": "num_ctx  32768",
            "model_info": {"qwen3.context_length": 262144},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_cls.return_value.__exit__ = MagicMock(return_value=None)

            result = _query_ollama_num_ctx("http://localhost:11434", "qwen3:4b")

        # parameters num_ctx (32768) should be returned, not arch (262144)
        assert result == 32768
