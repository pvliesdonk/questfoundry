"""Tests for providers.model_info module."""

from __future__ import annotations

from unittest.mock import patch

from questfoundry.providers.model_info import (
    ModelInfo,
    ModelProperties,
    get_capability_tier,
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


class TestModelPropertiesCapabilityTier:
    """Tests for capability_tier field on ModelProperties."""

    def test_default_tier_is_small(self) -> None:
        """ModelProperties defaults to small capability tier."""
        props = ModelProperties(context_window=32_768)
        assert props.capability_tier == "small"

    def test_explicit_large_tier(self) -> None:
        """ModelProperties accepts explicit large tier."""
        props = ModelProperties(context_window=128_000, capability_tier="large")
        assert props.capability_tier == "large"

    def test_ollama_models_are_small(self) -> None:
        """All Ollama models in KNOWN_MODELS are small tier."""
        from questfoundry.providers.model_info import KNOWN_MODELS

        for model_name, props in KNOWN_MODELS["ollama"].items():
            assert props.capability_tier == "small", f"ollama/{model_name} should be small"

    def test_cloud_models_are_large(self) -> None:
        """All cloud provider models in KNOWN_MODELS are large tier."""
        from questfoundry.providers.model_info import KNOWN_MODELS

        for provider in ("openai", "anthropic", "google"):
            for model_name, props in KNOWN_MODELS[provider].items():
                assert props.capability_tier == "large", f"{provider}/{model_name} should be large"


class TestGetCapabilityTier:
    """Tests for get_capability_tier() function."""

    def test_known_ollama_model(self) -> None:
        """Known Ollama model returns small."""
        assert get_capability_tier("ollama", "qwen3:4b-instruct-32k") == "small"

    def test_known_openai_model(self) -> None:
        """Known OpenAI model returns large."""
        assert get_capability_tier("openai", "gpt-4o") == "large"

    def test_known_anthropic_model(self) -> None:
        """Known Anthropic model returns large."""
        assert get_capability_tier("anthropic", "claude-sonnet-4-20250514") == "large"

    def test_known_google_model(self) -> None:
        """Known Google model returns large."""
        assert get_capability_tier("google", "gemini-2.5-pro") == "large"

    def test_unknown_ollama_small_pattern(self) -> None:
        """Unknown Ollama model with small name pattern returns small."""
        assert get_capability_tier("ollama", "phi3:3.8b") == "small"

    def test_unknown_ollama_large_pattern(self) -> None:
        """Unknown Ollama model without small pattern defaults to small."""
        assert get_capability_tier("ollama", "deepseek-r1:70b") == "small"

    def test_unknown_cloud_model_defaults_large(self) -> None:
        """Unknown cloud provider model defaults to large."""
        assert get_capability_tier("openai", "gpt-6-turbo") == "large"

    def test_unknown_provider_defaults_small(self) -> None:
        """Unknown provider defaults to small (conservative)."""
        assert get_capability_tier("local_llm", "custom-model") == "small"

    def test_case_insensitive_provider(self) -> None:
        """Provider name is case-insensitive."""
        assert get_capability_tier("OpenAI", "gpt-4o") == "large"
        assert get_capability_tier("OLLAMA", "qwen3:4b-instruct-32k") == "small"

    def test_heuristic_detects_7b(self) -> None:
        """Name pattern heuristic detects 7b as small."""
        assert get_capability_tier("ollama", "mistral-nemo:7b-q4") == "small"

    def test_heuristic_detects_13b(self) -> None:
        """Name pattern heuristic detects 13b as small."""
        assert get_capability_tier("ollama", "codellama:13b") == "small"

    def test_cloud_mini_is_large(self) -> None:
        """Cloud 'mini' models are large (trained to follow instructions well)."""
        assert get_capability_tier("openai", "gpt-5-mini") == "large"
        assert get_capability_tier("openai", "o3-mini") == "large"
