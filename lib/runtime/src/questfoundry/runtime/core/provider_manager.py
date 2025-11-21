"""
Provider Manager - Multi-provider LLM configuration and selection.

Supports: Anthropic, OpenAI, Google AI Studio, Ollama, LiteLLM

Uses capability-based model tiers (flagship, fast, reasoning, vision, long-context)
that map to provider-specific models via configuration file.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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

    def __init__(self, tier_config_path: Optional[Path] = None):
        """
        Initialize provider manager.

        Args:
            tier_config_path: Path to model_tiers.yaml (auto-detected if not provided)
        """
        self.available_providers = self._detect_providers()
        self.tier_mapping = self._load_tier_mapping(tier_config_path)
        logger.info(f"Detected providers: {', '.join(self.available_providers) or 'none'}")

    def _load_tier_mapping(self, tier_config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load model tier mapping from YAML configuration.

        Args:
            tier_config_path: Path to model_tiers.yaml

        Returns:
            Tier mapping dict
        """
        if tier_config_path is None:
            # Auto-detect: look for config/model_tiers.yaml
            runtime_dir = Path(__file__).parent.parent
            tier_config_path = runtime_dir / "config" / "model_tiers.yaml"

        # Also check environment variable override
        env_path = os.getenv("QF_MODEL_TIERS_CONFIG")
        if env_path:
            tier_config_path = Path(env_path)

        try:
            with open(tier_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded model tiers from: {tier_config_path}")
                return config
        except FileNotFoundError:
            logger.warning(
                f"Model tier config not found: {tier_config_path}. "
                "Using fallback defaults."
            )
            return self._get_fallback_tier_mapping()
        except Exception as e:
            logger.error(f"Failed to load tier config: {e}. Using fallback defaults.")
            return self._get_fallback_tier_mapping()

    def _get_fallback_tier_mapping(self) -> Dict[str, Any]:
        """Get fallback tier mapping when config file is not available."""
        return {
            "model_tiers": {
                "creative-writing": {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                    "ollama": "llama3.1:70b",
                    "litellm": "gpt-4o",
                },
                "structured-thinking": {
                    "anthropic": "claude-3-opus-20240229",
                    "openai": "o1-preview",
                    "google": "gemini-1.5-pro",
                    "ollama": "deepseek-r1:70b",
                    "litellm": "o1-preview",
                },
                "validation": {
                    "anthropic": "claude-3-5-haiku-20241022",
                    "openai": "gpt-4o-mini",
                    "google": "gemini-1.5-flash",
                    "ollama": "llama3.1:8b",
                    "litellm": "gpt-3.5-turbo",
                },
                "customer-facing": {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                    "ollama": "llama3.1:70b",
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

    def _detect_providers(self) -> List[str]:
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

    def select_provider(
        self,
        preferred_provider: Optional[str] = None,
        fallback_chain: Optional[List[str]] = None
    ) -> str:
        """
        Select best available provider.

        Args:
            preferred_provider: Preferred provider name (or "auto")
            fallback_chain: Ordered list of fallback providers

        Returns:
            Selected provider name

        Raises:
            RuntimeError: If no providers are available
        """
        if not self.available_providers:
            raise RuntimeError(
                "No LLM providers available. Set at least one API key: "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or configure Ollama/LiteLLM"
            )

        # If "auto" or None, use first available
        if preferred_provider in (None, "auto"):
            selected = self.available_providers[0]
            logger.info(f"Auto-selected provider: {selected}")
            return selected

        # Try preferred provider
        if preferred_provider in self.available_providers:
            logger.info(f"Using preferred provider: {preferred_provider}")
            return preferred_provider

        # Try fallback chain
        if fallback_chain:
            for fallback in fallback_chain:
                if fallback in self.available_providers:
                    logger.warning(
                        f"Preferred provider '{preferred_provider}' unavailable. "
                        f"Using fallback: {fallback}"
                    )
                    return fallback

        # Last resort: use first available
        selected = self.available_providers[0]
        logger.warning(
            f"Provider '{preferred_provider}' and fallbacks unavailable. "
            f"Using: {selected}"
        )
        return selected

    def get_recommended_tier(self, role_id: str) -> str:
        """
        Get recommended model tier for a role.

        Args:
            role_id: Role identifier (e.g., "showrunner", "plotwright")

        Returns:
            Recommended tier name

        Examples:
            >>> get_recommended_tier("showrunner")
            "customer-facing"

            >>> get_recommended_tier("scene_smith")
            "creative-writing"
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
            model_spec: Either a tier name (e.g., "creative-writing") or specific model (e.g., "gpt-4o")

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
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Any:
        """
        Create LangChain LLM client for the specified provider.

        Args:
            provider: Provider name
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            LangChain chat model instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider == "anthropic":
            return self._create_anthropic_client(model, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            return self._create_openai_client(model, temperature, max_tokens, **kwargs)
        elif provider == "google":
            return self._create_google_client(model, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            return self._create_ollama_client(model, temperature, max_tokens, **kwargs)
        elif provider == "litellm":
            return self._create_litellm_client(model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_anthropic_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Anthropic client."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install langchain-anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=api_key,
            **kwargs
        )

    def _create_openai_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create OpenAI client."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            **kwargs
        )

    def _create_google_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Google AI Studio client."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Install with: pip install langchain-google-genai"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
            **kwargs
        )

    def _create_ollama_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create Ollama client (local)."""
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Install with: pip install langchain-community"
            )

        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=base_url,
            **kwargs
        )

    def _create_litellm_client(self, model: str, temperature: float, max_tokens: int, **kwargs):
        """Create LiteLLM client (proxy)."""
        try:
            from langchain_community.chat_models import ChatLiteLLM
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Install with: pip install langchain-community"
            )

        api_base = os.getenv("LITELLM_API_BASE")
        if not api_base:
            raise ValueError("LITELLM_API_BASE environment variable not set")

        api_key = os.getenv("LITELLM_API_KEY", "sk-1234")  # LiteLLM may not require key

        return ChatLiteLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            **kwargs
        )

    def get_config_summary(self) -> Dict[str, Any]:
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
