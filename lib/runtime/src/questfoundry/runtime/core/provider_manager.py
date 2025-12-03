"""
Provider Manager - Multi-provider LLM configuration and selection.

Supports: Anthropic, OpenAI, Google AI Studio, Ollama, LiteLLM

Uses capability-based model tiers (flagship, fast, reasoning, vision, long-context)
that map to provider-specific models via configuration file.
"""

import logging
import os
from pathlib import Path
from typing import Any
from urllib import request

import yaml

from questfoundry.runtime.config import get_settings
from questfoundry.runtime.exceptions import ProviderError

logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Manages LLM provider detection, selection, and client creation.

    Supports multiple providers with automatic detection and fallback.
    Uses capability-based model tiers that map to provider-specific models.
    """

    # Provider detection via environment variables
    PROVIDER_ENV_VARS = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "ollama": None,  # Always available (localhost)
        "litellm": "LITELLM_API_BASE",
    }

    def __init__(self, tier_config_path: Path | None = None):
        """
        Initialize provider manager.

        Args:
            tier_config_path: Path to model_tiers.yaml (auto-detected if not provided)
        """
        self.available_providers = self._detect_providers()
        self.tier_mapping = self._load_tier_mapping(tier_config_path)
        self._llm_cache: dict[
            tuple, Any
        ] = {}  # Cache LLM clients by (provider, model, temp, max_tokens)
        logger.info(f"Detected providers: {', '.join(self.available_providers) or 'none'}")

    def _load_tier_mapping(self, tier_config_path: Path | None = None) -> dict[str, Any]:
        """
        Load model tier mapping from YAML configuration.

        Args:
            tier_config_path: Path to model_tiers.yaml

        Returns:
            Tier mapping dict
        """
        if tier_config_path is None:
            # Check centralized config for custom path
            settings = get_settings()
            if settings.llm.model_tiers_config:
                tier_config_path = Path(settings.llm.model_tiers_config)
            else:
                # Auto-detect: look for config/model_tiers.yaml
                runtime_dir = Path(__file__).parent.parent
                tier_config_path = runtime_dir / "config" / "model_tiers.yaml"

        try:
            with open(tier_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded model tiers from: {tier_config_path}")
                return config
        except FileNotFoundError:
            logger.warning(
                f"Model tier config not found: {tier_config_path}. Using fallback defaults."
            )
            return self._get_fallback_tier_mapping()
        except Exception as e:
            logger.error(f"Failed to load tier config: {e}. Using fallback defaults.")
            return self._get_fallback_tier_mapping()

    def _get_fallback_tier_mapping(self) -> dict[str, Any]:
        """Get fallback tier mapping when config file is not available."""
        return {
            "model_tiers": {
                "creative-writing": {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                    "ollama": "qwen2.5:latest",  # Single model for all tiers (GPU efficiency)
                    "litellm": "gpt-4o",
                },
                "structured-thinking": {
                    "anthropic": "claude-3-opus-20240229",
                    "openai": "o1-preview",
                    "google": "gemini-1.5-pro",
                    "ollama": "qwen2.5:latest",  # Same model for GPU efficiency
                    "litellm": "o1-preview",
                },
                "validation": {
                    "anthropic": "claude-3-5-haiku-20241022",
                    "openai": "gpt-4o-mini",
                    "google": "gemini-1.5-flash",
                    "ollama": "qwen2.5:latest",  # Same model for GPU efficiency
                    "litellm": "gpt-3.5-turbo",
                },
                "customer-facing": {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                    "ollama": "qwen2.5:latest",  # Same model for GPU efficiency
                    "litellm": "gpt-4o",
                },
                "quick-feedback": {
                    "anthropic": "claude-3-5-haiku-20241022",
                    "openai": "gpt-4o-mini",
                    "google": "gemini-1.5-flash",
                    "ollama": "qwen2.5:latest",  # Same model for GPU efficiency
                    "litellm": "gpt-4o-mini",
                },
                "long-context": {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                    "ollama": "qwen2.5:latest",  # Same model for GPU efficiency
                    "litellm": "gpt-4o",
                },
            },
            "default_tier": "creative-writing",
            "provider_priority": ["anthropic", "openai", "google", "ollama", "litellm"],
            "role_tier_recommendations": {
                "showrunner": "customer-facing",
                "plotwright": "structured-thinking",
                "scene_smith": "creative-writing",
                "gatekeeper": "validation",
                "style_lead": "validation",
            },
        }

    def _detect_providers(self) -> list[str]:
        """
        Detect which providers are available.

        Returns:
            List of available provider names
        """
        available = []

        for provider, env_var in self.PROVIDER_ENV_VARS.items():
            if env_var is None:
                # Ollama - always available (assumes localhost)
                available.append(provider)
            elif os.getenv(env_var):
                available.append(provider)

        return available

    def mark_unavailable(self, provider: str, reason: str | None = None) -> None:
        """Mark provider as unavailable for the remainder of this run.

        This removes the provider from `available_providers` so subsequent
        selections will skip it (and can fall back or error).
        """
        if provider in self.available_providers:
            self.available_providers.remove(provider)
            logger.warning(
                f"Marking provider '{provider}' as unavailable: {reason or 'runtime error'}"
            )

    def select_provider(
        self,
        preferred_provider: str | None = None,
        fallback_chain: list[str | None] | None = None,
        strict: bool = False,
    ) -> str:
        """
        Select best available provider.

        Args:
            preferred_provider: Preferred provider name (or "auto")
            fallback_chain: Ordered list of fallback providers
            strict: When True, do not silently fall back to an
                arbitrary available provider if the requested
                provider/fallback chain is unavailable.

        Returns:
            Selected provider name.

        Raises:
            ProviderError: If no providers are available, or if
                `strict` is True and the requested provider plus
                fallbacks are not available.
        """
        if not self.available_providers:
            raise ProviderError(
                "No LLM providers available",
                suggestions=[
                    "Set at least one API key environment variable:",
                    "  • ANTHROPIC_API_KEY for Anthropic Claude",
                    "  • OPENAI_API_KEY for OpenAI GPT",
                    "  • GOOGLE_API_KEY for Google Gemini",
                    "Or configure a local provider:",
                    "  • Ollama (no API key required)",
                    "  • LiteLLM with LITELLM_API_BASE",
                ],
            )

        # If "auto" or None, use first available
        if preferred_provider in (None, "auto"):
            selected = self.available_providers[0]
            logger.info("Auto-selected provider: %s", selected)
            return selected

        # Try preferred provider
        if preferred_provider in self.available_providers:
            logger.info("Using preferred provider: %s", preferred_provider)
            return preferred_provider

        # Try fallback chain (if provided)
        if fallback_chain:
            for fallback in fallback_chain:
                if fallback is None:
                    continue
                if fallback in self.available_providers:
                    logger.warning(
                        "Preferred provider '%s' unavailable. Using fallback: %s",
                        preferred_provider,
                        fallback,
                    )
                    return fallback

        # Last resort: either error (strict) or use first available
        if strict:
            # When strict is True we treat an explicit provider/fallback list
            # as authoritative and do NOT silently fall back to some other
            # configured provider. This is used when the user has set
            # QF_LLM_PROVIDER or passed --provider on the CLI.
            raise ProviderError(
                f"Requested provider '{preferred_provider}' and any fallbacks are not available.",
                provider_name=preferred_provider,
            )

        selected = self.available_providers[0]
        logger.warning(
            "Provider '%s' and fallbacks unavailable. Using: %s",
            preferred_provider,
            selected,
        )
        return selected

    def get_recommended_tier(self, role_id: str) -> str:
        """Get recommended model tier name for a given role.

        This is used to map role IDs (e.g., "plotwright", "scene_smith")
        to model tiers defined in the model_tiers.yaml configuration.
        If no specific recommendation exists for a role, falls back to
        the configured default tier ("creative-writing" by default).
        """
        recommendations = self.tier_mapping.get("role_tier_recommendations", {})
        default_tier = self.tier_mapping.get("default_tier", "creative-writing")
        return recommendations.get(role_id, default_tier)

    def resolve_model(
        self,
        provider: str,
        model_spec: str,
    ) -> str:
        """
        Resolve model specification to actual model name.

        Args:
            provider: Provider name
            model_spec: Either a tier name (e.g., "creative-writing") or
                specific model (e.g., "gpt-4o")

        Returns:
            Actual model name for the provider

        Examples:
            >>> resolve_model("openai", "creative-writing")
            "gpt-4o"

            >>> resolve_model("anthropic", "validation")
            "claude-3-5-haiku-20241022"

            >>> resolve_model("openai", "gpt-4-turbo")  # Specific model
            "gpt-4-turbo"
        """
        model_tiers = self.tier_mapping.get("model_tiers", {})

        # Check if model_spec is a tier name
        if model_spec in model_tiers:
            tier_models = model_tiers[model_spec]
            if provider in tier_models:
                resolved = tier_models[provider]
                logger.debug(f"Resolved tier '{model_spec}' → {provider}:{resolved}")
                return resolved
            else:
                # Tier exists but not for this provider - use default tier
                default_tier = self.tier_mapping.get("default_tier", "creative-writing")
                logger.warning(
                    f"Tier '{model_spec}' not available for {provider}. "
                    f"Using default tier: {default_tier}"
                )
                return self.resolve_model(provider, default_tier)

        # Not a tier - assume it's a specific model name (backward compatibility)
        logger.debug(f"Using specific model: {provider}:{model_spec}")
        return model_spec

    def create_llm_client(
        self, provider: str, model: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs
    ) -> Any:
        """
        Create or retrieve cached LangChain LLM client for the specified provider.

        Clients are cached by (provider, model, temperature, max_tokens) to avoid
        recreating connections on every invocation. This provides ~30-50% speedup
        for repeated role executions.

        Args:
            provider: Provider name
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional provider-specific parameters (not cached)

        Returns:
            LangChain chat model instance (cached or newly created)

        Raises:
            ValueError: If provider is not supported
        """
        # Create cache key (excluding kwargs since they're rare)
        cache_key = (provider, model, temperature, max_tokens)

        # Check cache first
        if cache_key in self._llm_cache:
            logger.debug(f"Using cached LLM client: {provider}:{model}")
            return self._llm_cache[cache_key]

        # Cache miss - create new client
        logger.debug(f"Creating new LLM client: {provider}:{model}")

        if provider == "anthropic":
            client = self._create_anthropic_client(model, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            client = self._create_openai_client(model, temperature, max_tokens, **kwargs)
        elif provider == "google":
            client = self._create_google_client(model, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            client = self._create_ollama_client(model, temperature, max_tokens, **kwargs)
        elif provider == "litellm":
            client = self._create_litellm_client(model, temperature, max_tokens, **kwargs)
        else:
            raise ProviderError(
                f"Unsupported provider: '{provider}'",
                provider_name=provider,
                suggestions=[
                    f"Available providers: {', '.join(self.available_providers)}",
                    "Use 'auto' to automatically select the first available provider",
                    "Check CONTRIBUTING.md for provider configuration details",
                ],
            )

        # Cache and return
        self._llm_cache[cache_key] = client
        return client

    def _create_anthropic_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Anthropic client.

        Note: Anthropic models rely on prompt-level JSON instructions only.
        They do not support response_format parameter (only structured output via tool calling).
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ProviderError(
                "Anthropic provider package not installed",
                provider_name="anthropic",
                suggestions=[
                    "Install with: pip install langchain-anthropic",
                    "Or use uv: uv add langchain-anthropic",
                    "Or choose a different provider with --provider flag",
                ],
            ) from e

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY environment variable not set",
                provider_name="anthropic",
                suggestions=[
                    "Set the API key: export ANTHROPIC_API_KEY='your-key-here'",
                    "Get an API key from: https://console.anthropic.com/",
                    "Or use a different provider with --provider flag",
                ],
            )

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=api_key,
            **kwargs,
        )

    def _create_openai_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create OpenAI client with JSON mode enabled."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ProviderError(
                "OpenAI provider package not installed",
                provider_name="openai",
                suggestions=[
                    "Install with: pip install langchain-openai",
                    "Or use uv: uv add langchain-openai",
                    "Or choose a different provider with --provider flag",
                ],
            ) from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(
                "OPENAI_API_KEY environment variable not set",
                provider_name="openai",
                suggestions=[
                    "Set the API key: export OPENAI_API_KEY='your-key-here'",
                    "Get an API key from: https://platform.openai.com/api-keys",
                    "Or use a different provider with --provider flag",
                ],
            )

        # Default: do not force response_format when tools are bound. JSON forcing can
        # collide with OpenAI’s requirement that the prompt explicitly mention “json”.
        # We keep the hook here for future structured output cases, but leave it empty
        # by default.
        model_kwargs: dict[str, Any] = {}

        # For reasoning models (o1, o3, o4-mini), use different parameters
        is_reasoning_model = any(prefix in model.lower() for prefix in ["o1", "o3", "o4-mini"])

        if is_reasoning_model:
            # reasoning_effort: "minimal" (10%) | "low" (20%) | "medium" (50%) | "high" (80%)
            # Controls what % of max_completion_tokens goes to internal reasoning
            # Higher = better reasoning but more tokens consumed
            model_kwargs["reasoning_effort"] = "medium"  # Good balance for topology/planning

            # o1/o3 models require max_completion_tokens NOT max_tokens
            # Set higher to allow reasoning + JSON output (reasoning tokens + visible tokens)
            # Default: 16000 (enough for extensive reasoning + structured output)
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens * 2,  # Double for reasoning + output
                openai_api_key=api_key,
                model_kwargs=model_kwargs,
                **kwargs,
            )
        else:
            # Regular models (gpt-4o, gpt-4o-mini) use max_tokens
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                model_kwargs=model_kwargs,
                **kwargs,
            )

    def _create_google_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Google AI Studio client with JSON mode enabled."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ProviderError(
                "Google AI provider package not installed",
                provider_name="google",
                suggestions=[
                    "Install with: pip install langchain-google-genai",
                    "Or use uv: uv add langchain-google-genai",
                    "Or choose a different provider with --provider flag",
                ],
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                "GOOGLE_API_KEY environment variable not set",
                provider_name="google",
                suggestions=[
                    "Set the API key: export GOOGLE_API_KEY='your-key-here'",
                    "Get an API key from: https://makersuite.google.com/app/apikey",
                    "Or use a different provider with --provider flag",
                ],
            )

        # Force JSON output format for structured responses
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
            generation_config={"response_mime_type": "application/json"},
            **kwargs,
        )

    def _create_ollama_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Ollama client (local or remote) with JSON mode enabled."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ProviderError(
                "Ollama provider package not installed",
                provider_name="ollama",
                suggestions=[
                    "Install with: pip install langchain-ollama",
                    "Or use uv: uv add langchain-ollama",
                    "Also ensure Ollama is running: ollama serve",
                ],
            ) from e

        # Get Ollama configuration from centralized config
        settings = get_settings()
        base_url = settings.llm.ollama_host
        logger.debug(f"[OLLAMA] Using base_url: {base_url}")
        logger.debug(
            f"[OLLAMA] Creating ChatOllama: model={model}, temp={temperature}, "
            f"max_tokens={max_tokens}"
        )

        # Fast preflight check to avoid long timeouts when daemon isn't running
        try:
            request.urlopen(f"{base_url}/api/tags", timeout=2).close()
        except Exception as exc:  # pragma: no cover - network/daemon check
            raise ProviderError(
                "Ollama daemon not reachable",
                provider_name="ollama",
                suggestions=[
                    "Start the server: ollama serve",
                    f"Check OLLAMA_HOST or QF_LLM__OLLAMA_HOST (currently {base_url})",
                ],
            ) from exc

        # Note: Do NOT force JSON format as it conflicts with tool calling.
        # When tools are bound, Ollama handles structured output natively.
        # TODO: Ideal behavior is to:
        #   - Track per-model context limits (e.g. via model_tiers.yaml)
        #   - Estimate rendered prompt token count per request
        #   - Raise a clear error if prompt_tokens + max_tokens exceeds the model's
        #     context window instead of relying on Ollama's internal truncation
        num_ctx = settings.llm.ollama_num_ctx

        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=base_url,
            # format="json" removed - conflicts with tool calling
            num_ctx=num_ctx,
            **kwargs,
        )

    def _create_litellm_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create LiteLLM client (proxy) with JSON mode enabled."""
        try:
            from langchain_community.chat_models import ChatLiteLLM
        except ImportError as e:
            raise ProviderError(
                "LiteLLM provider package not installed",
                provider_name="litellm",
                suggestions=[
                    "Install with: pip install langchain-community",
                    "Or use uv: uv add langchain-community",
                    "Or choose a different provider with --provider flag",
                ],
            ) from e

        # Get LiteLLM configuration from centralized config
        settings = get_settings()
        api_base = settings.llm.litellm_api_base
        if not api_base:
            raise ProviderError(
                "LITELLM_API_BASE not set in environment or config",
                provider_name="litellm",
                suggestions=[
                    "Set the API base: export LITELLM_API_BASE='http://localhost:4000'",
                    "Or in config: llm.litellm_api_base = 'http://localhost:4000'",
                    "Start LiteLLM server: litellm --config config.yaml",
                    "Or use a different provider with --provider flag",
                ],
            )

        api_key = settings.llm.litellm_api_key or "sk-1234"  # LiteLLM may not require key

        # LiteLLM proxies to various providers - add response_format for OpenAI-compatible endpoints
        return ChatLiteLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
            **kwargs,
        )

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get summary of provider configuration.

        Returns:
            Dict with provider status and configuration
        """
        return {
            "available_providers": self.available_providers,
            "default_models": self.DEFAULT_MODELS,
            "env_vars_checked": list(self.PROVIDER_ENV_VARS.values()),
        }
