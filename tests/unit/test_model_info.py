"""Tests for providers.model_info module."""

from __future__ import annotations

from unittest.mock import patch

from questfoundry.providers.model_info import ModelInfo, get_model_info


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
