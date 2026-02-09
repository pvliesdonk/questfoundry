"""Tests for providers.model_info module."""

from __future__ import annotations

from unittest.mock import patch

from questfoundry.providers.model_info import (
    KNOWN_MODELS,
    ModelInfo,
    ModelProperties,
    get_model_info,
)


class TestModelInfoDefaults:
    """Tests for max_concurrency provider defaults."""

    def test_ollama_concurrency(self) -> None:
        """Ollama models get low concurrency (local GPU)."""
        info = get_model_info("ollama", "qwen3:4b-instruct-32k")
        assert info.max_concurrency == 2

    def test_openai_concurrency(self) -> None:
        """OpenAI models get high concurrency (cloud API)."""
        info = get_model_info("openai", "gpt-5-mini")
        assert info.max_concurrency == 20

    def test_anthropic_concurrency(self) -> None:
        """Anthropic models get moderate concurrency."""
        info = get_model_info("anthropic", "claude-sonnet-4-20250514")
        assert info.max_concurrency == 10

    def test_google_concurrency(self) -> None:
        """Google models get high concurrency (cloud API)."""
        info = get_model_info("google", "gemini-2.5-flash")
        assert info.max_concurrency == 20

    def test_unknown_provider_concurrency(self) -> None:
        """Unknown providers fall back to concurrency of 2."""
        info = get_model_info("unknown_provider", "some-model")
        assert info.max_concurrency == 2

    def test_env_var_override(self) -> None:
        """QF_MAX_CONCURRENCY env var overrides provider default."""
        with patch.dict("os.environ", {"QF_MAX_CONCURRENCY": "5"}):
            info = get_model_info("openai", "gpt-5-mini")
            assert info.max_concurrency == 5

    def test_env_var_override_for_ollama(self) -> None:
        """QF_MAX_CONCURRENCY overrides even low-concurrency providers."""
        with patch.dict("os.environ", {"QF_MAX_CONCURRENCY": "10"}):
            info = get_model_info("ollama", "qwen3:4b-instruct-32k")
            assert info.max_concurrency == 10


class TestModelInfoDataclass:
    """Tests for ModelInfo dataclass."""

    def test_default_max_concurrency(self) -> None:
        """ModelInfo defaults to max_concurrency=2."""
        info = ModelInfo(context_window=32_768)
        assert info.max_concurrency == 2

    def test_custom_max_concurrency(self) -> None:
        """ModelInfo accepts custom max_concurrency."""
        info = ModelInfo(context_window=32_768, max_concurrency=15)
        assert info.max_concurrency == 15

    def test_frozen(self) -> None:
        """ModelInfo is frozen (immutable)."""
        info = ModelInfo(context_window=32_768, max_concurrency=5)
        with __import__("pytest").raises(AttributeError):
            info.max_concurrency = 10  # type: ignore[misc]


class TestModelPropertiesCapabilityFlags:
    """Tests for supports_verbosity and supports_reasoning_effort flags."""

    def test_gpt5_mini_supports_both(self) -> None:
        """GPT-5-mini supports both verbosity and reasoning_effort."""
        props = KNOWN_MODELS["openai"]["gpt-5-mini"]
        assert props.supports_verbosity is True
        assert props.supports_reasoning_effort is True

    def test_gpt5_family_supports_both(self) -> None:
        """All GPT-5 family models support verbosity and reasoning_effort."""
        for model in ("gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2"):
            props = KNOWN_MODELS["openai"][model]
            assert props.supports_verbosity is True, f"{model} should support verbosity"
            assert props.supports_reasoning_effort is True, (
                f"{model} should support reasoning_effort"
            )

    def test_o1_supports_reasoning_only(self) -> None:
        """o1 supports reasoning_effort but not verbosity."""
        props = KNOWN_MODELS["openai"]["o1"]
        assert props.supports_reasoning_effort is True
        assert props.supports_verbosity is False

    def test_o3_mini_supports_reasoning_only(self) -> None:
        """o3-mini supports reasoning_effort but not verbosity."""
        props = KNOWN_MODELS["openai"]["o3-mini"]
        assert props.supports_reasoning_effort is True
        assert props.supports_verbosity is False

    def test_o4_mini_supports_reasoning_only(self) -> None:
        """o4-mini supports reasoning_effort but not verbosity."""
        props = KNOWN_MODELS["openai"]["o4-mini"]
        assert props.supports_reasoning_effort is True
        assert props.supports_verbosity is False

    def test_gpt4o_no_special_support(self) -> None:
        """GPT-4o does not support verbosity or reasoning_effort."""
        props = KNOWN_MODELS["openai"]["gpt-4o"]
        assert props.supports_verbosity is False
        assert props.supports_reasoning_effort is False

    def test_gpt41_family_no_special_support(self) -> None:
        """GPT-4.1 family does not support verbosity or reasoning_effort."""
        for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            props = KNOWN_MODELS["openai"][model]
            assert props.supports_verbosity is False, f"{model}"
            assert props.supports_reasoning_effort is False, f"{model}"

    def test_ollama_models_no_special_support(self) -> None:
        """Ollama models default to no verbosity/reasoning_effort support."""
        props = KNOWN_MODELS["ollama"]["qwen3:4b-instruct-32k"]
        assert props.supports_verbosity is False
        assert props.supports_reasoning_effort is False

    def test_anthropic_models_no_special_support(self) -> None:
        """Anthropic models default to no verbosity/reasoning_effort support."""
        props = KNOWN_MODELS["anthropic"]["claude-sonnet-4-20250514"]
        assert props.supports_verbosity is False
        assert props.supports_reasoning_effort is False

    def test_model_properties_default_flags(self) -> None:
        """ModelProperties defaults both flags to False."""
        props = ModelProperties(context_window=32_768)
        assert props.supports_verbosity is False
        assert props.supports_reasoning_effort is False


class TestModelRegistryContextWindows:
    """Tests for correct context window values in KNOWN_MODELS."""

    def test_gpt5_mini_context_window(self) -> None:
        """GPT-5-mini has 400K context window."""
        assert KNOWN_MODELS["openai"]["gpt-5-mini"].context_window == 400_000

    def test_gpt41_family_context_window(self) -> None:
        """GPT-4.1 family has 1M context window."""
        for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            assert KNOWN_MODELS["openai"][model].context_window == 1_000_000, f"{model}"

    def test_qwen25_7b_context_window(self) -> None:
        """qwen2.5:7b has 128K context window (not 32K)."""
        assert KNOWN_MODELS["ollama"]["qwen2.5:7b"].context_window == 128_000

    def test_retired_models_removed(self) -> None:
        """Retired models are no longer in the registry."""
        assert "o1-mini" not in KNOWN_MODELS["openai"]
        assert "o1-preview" not in KNOWN_MODELS["openai"]
        assert "gpt-4" not in KNOWN_MODELS["openai"]
        assert "gpt-4-turbo" not in KNOWN_MODELS["openai"]
        assert "gpt-3.5-turbo" not in KNOWN_MODELS["openai"]
        assert "claude-3-5-sonnet-latest" not in KNOWN_MODELS["anthropic"]
        assert "claude-3-5-sonnet-20241022" not in KNOWN_MODELS["anthropic"]
        assert "claude-3-opus-20240229" not in KNOWN_MODELS["anthropic"]
        assert "claude-3-haiku-20240307" not in KNOWN_MODELS["anthropic"]

    def test_new_models_present(self) -> None:
        """New models are present in the registry."""
        # OpenAI
        for model in ("gpt-5", "gpt-5-nano", "gpt-5.1", "gpt-5.2", "o3-pro", "o4-mini"):
            assert model in KNOWN_MODELS["openai"], f"{model} missing"
        # Anthropic
        for model in (
            "claude-opus-4-6",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ):
            assert model in KNOWN_MODELS["anthropic"], f"{model} missing"
        # Google
        for model in ("gemini-2.5-flash-lite", "gemini-3-pro-preview", "gemini-3-flash-preview"):
            assert model in KNOWN_MODELS["google"], f"{model} missing"
