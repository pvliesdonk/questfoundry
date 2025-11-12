"""Abstract base classes for QuestFoundry providers"""

import logging
import types
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from questfoundry.providers.cache import (
    CacheConfig,
    ResponseCache,
    generate_cache_key,
)
from questfoundry.providers.rate_limiter import (
    CostTracker,
    RateLimitConfig,
    RateLimiter,
)

logger = logging.getLogger(__name__)


class Provider(ABC):
    """
    Base class for all QuestFoundry providers.

    Providers enable integration with external AI services for text generation,
    image generation, audio synthesis, and other creative AI capabilities.
    They abstract away API specifics and provide a unified interface for
    QuestFoundry roles to leverage external AI models.

    Provider types:
        - Text providers: LLM text generation (OpenAI, Anthropic, Ollama, etc.)
        - Image providers: Image generation (DALL-E, Stable Diffusion, etc.)
        - Audio providers: Text-to-speech and audio generation (ElevenLabs, etc.)

    Built-in features:
        - Response caching: Avoid redundant API calls and reduce costs
        - Rate limiting: Respect provider API limits and manage concurrency
        - Cost tracking: Monitor API usage and costs
        - Retry logic: Handle transient failures automatically
        - Streaming support: Enable streaming responses where available

    Provider plugin pattern:
        New providers can be added by:
        1. Subclassing Provider (or TextProvider, ImageProvider, AudioProvider)
        2. Implementing required abstract methods
        3. Registering in the provider registry
        4. Configuring via project settings

    Configuration:
        Providers are configured via dictionaries with common keys:
        - cache_enabled: Enable response caching (default: True)
        - cache_ttl_seconds: Cache time-to-live in seconds
        - rate_limit_config: Rate limiting parameters
        - Provider-specific keys (api_key, base_url, model, etc.)

    Example provider usage:
        >>> from questfoundry.providers.text.openai import OpenAIProvider
        >>> config = {
        ...     "api_key": "sk-...",
        ...     "model": "gpt-4",
        ...     "cache_enabled": True
        ... }
        >>> provider = OpenAIProvider(config)
        >>> response = provider.generate_text("Write a tavern scene")
        >>> print(response)

    Implementing a custom provider:
        >>> from questfoundry.providers.base import Provider
        >>> class MyCustomProvider(Provider):
        ...     def __init__(self, config):
        ...         super().__init__(config)
        ...         self.api_key = config["api_key"]
        ...
        ...     def validate_config(self):
        ...         if not self.api_key:
        ...             raise ValueError("api_key required")
        ...
        ...     # Implement other abstract methods...
    """

    cache: Optional[ResponseCache]
    cache_config: Optional[CacheConfig]
    rate_limiter: Optional[RateLimiter]
    cost_tracker: Optional[CostTracker]

    def __init__(self, config: dict[str, Any]):
        """
        Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
                Can include:
                - 'cache_enabled': bool, enable response caching
                - 'cache_ttl_seconds': int, cache TTL
                - 'rate_limit_config': dict, rate limiting configuration
        """
        logger.debug("Initializing %s with configuration", self.__class__.__name__)
        logger.trace("Provider config keys: %s", list(config.keys()))
        self.config = config

        # Initialize cache if enabled
        cache_config_dict = config.get("cache", {})
        if cache_config_dict or config.get("cache_enabled", True):
            logger.debug("Initializing cache for %s", self.__class__.__name__)
            self.cache_config = (
                CacheConfig.from_dict(cache_config_dict)
                if cache_config_dict
                else CacheConfig()
            )
            self.cache = ResponseCache(
                cache_dir=self.cache_config.cache_dir,
                ttl_seconds=self.cache_config.ttl_seconds,
            )
            logger.trace(
                "Cache initialized with TTL=%s seconds", self.cache_config.ttl_seconds
            )
        else:
            logger.debug("Cache disabled for %s", self.__class__.__name__)
            self.cache = None
            self.cache_config = None

        # Initialize rate limiter if configured
        rate_limit_config_dict = config.get("rate_limit", {})
        if rate_limit_config_dict:
            logger.debug("Initializing rate limiter for %s", self.__class__.__name__)
            rate_config = RateLimitConfig(**rate_limit_config_dict)
            self.rate_limiter = RateLimiter(rate_config)
            self.cost_tracker = CostTracker()
            logger.trace(
                "Rate limiter configured with %d requests/minute",
                rate_config.requests_per_minute,
            )
        else:
            logger.trace("Rate limiter not configured for %s", self.__class__.__name__)
            self.rate_limiter = None
            self.cost_tracker = None

        logger.trace("%s initialization complete", self.__class__.__name__)

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate provider configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def close(self) -> None:
        """
        Close provider and release resources.

        Default implementation does nothing. Providers can override
        to cleanup connections, close clients, etc.
        """
        pass

    def __enter__(self) -> "Provider":
        """Context manager entry"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit"""
        self.close()

    def _get_cache_key(
        self,
        prompt: str,
        method: str = "text",
        **params: Any,
    ) -> str:
        """Generate cache key for a request.

        Args:
            prompt: Input prompt
            method: Method name ('text', 'image', 'audio')
            **params: Additional parameters (model, temperature, etc.)

        Returns:
            Cache key string
        """
        # Extract model without modifying params dictionary
        model = params.get("model", "default")
        # Create a copy excluding model to avoid duplication
        cache_params = {k: v for k, v in params.items() if k != "model"}
        return generate_cache_key(
            provider=self.__class__.__name__,
            model=model,
            prompt=prompt,
            **cache_params,
        )

    def _get_cached_response(self, key: str) -> Optional[str]:
        """Get cached response if available.

        Args:
            key: Cache key

        Returns:
            Cached response or None if not found/expired
        """
        if self.cache is None:
            return None
        cached = self.cache.get(key)
        if cached:
            logger.debug("Cache hit for key: %s", key)
        else:
            logger.trace("Cache miss for key: %s", key)
        return cached

    def _cache_response(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Cache a response.

        Args:
            key: Cache key
            value: Response to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if self.cache is None:
            return

        if ttl is None and self.cache_config:
            ttl = self.cache_config.ttl_seconds

        logger.trace("Caching response with key: %s (ttl=%s seconds)", key, ttl)
        self.cache.set(key, value, ttl=ttl)
        logger.debug("Response cached successfully")

    def _check_rate_limit(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        """Check if request would exceed rate limits.

        Args:
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            True if allowed, False if would exceed limits
        """
        if self.rate_limiter is None:
            return True

        logger.trace(
            "Checking rate limits: input_tokens=%d, output_tokens=%d",
            input_tokens,
            output_tokens,
        )
        allowed = self.rate_limiter.check_limit(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if not allowed:
            logger.warning("Rate limit exceeded for request")
        else:
            logger.trace("Request allowed within rate limits")
        return allowed

    def _record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_input_1k: float = 0.0,
        cost_per_output_1k: float = 0.0,
    ) -> None:
        """Record API usage for rate limiting and cost tracking.

        Args:
            model: Model name used
            input_tokens: Actual input tokens
            output_tokens: Actual output tokens
            cost_per_input_1k: Cost per 1000 input tokens
            cost_per_output_1k: Cost per 1000 output tokens
        """
        logger.trace(
            "Recording usage for model=%s: input=%d, output=%d tokens",
            model,
            input_tokens,
            output_tokens,
        )

        if self.rate_limiter:
            self.rate_limiter.record_usage(input_tokens, output_tokens)
            logger.trace("Rate limiter usage recorded")

        if self.cost_tracker:
            self.cost_tracker.record_request(
                provider=self.__class__.__name__,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_per_input_1k=cost_per_input_1k,
                cost_per_output_1k=cost_per_output_1k,
            )
            logger.trace("Cost tracking recorded")

        logger.debug("Usage recorded for model %s", model)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats, or empty dict if caching disabled
        """
        if self.cache is None:
            return {}
        return self.cache.get_stats()

    def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limit statistics.

        Returns:
            Dictionary with rate limit stats, or empty dict if disabled
        """
        if self.rate_limiter is None:
            return {}
        return self.rate_limiter.get_stats()

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost tracking summary.

        Returns:
            Dictionary with cost summary, or empty dict if disabled
        """
        if self.cost_tracker is None:
            return {}
        return self.cost_tracker.get_cost_summary()


class TextProvider(Provider):
    """
    Abstract base class for text generation providers.

    Text providers interface with LLMs to generate text based on prompts.
    Supports optional response caching and rate limiting.

    Example implementation with caching and rate limiting:

        class MyTextProvider(TextProvider):
            def generate_text(self, prompt: str, **kwargs):
                # 1. Generate cache key
                cache_key = self._get_cache_key(prompt, **kwargs)

                # 2. Check cache first
                cached = self._get_cached_response(cache_key)
                if cached:
                    return cached

                # 3. Check rate limits
                if not self._check_rate_limit(input_tokens=100):
                    raise RateLimitError("Rate limit exceeded")

                # 4. Make actual API call
                response = self._generate_text_uncached(prompt, **kwargs)

                # 5. Cache response
                self._cache_response(cache_key, response)

                # 6. Record usage
                self._record_usage(
                    model=kwargs.get("model", "default"),
                    input_tokens=100,
                    output_tokens=250,
                    cost_per_input_1k=0.03,
                    cost_per_output_1k=0.06,
                )

                return response

            def _generate_text_uncached(self, prompt: str, **kwargs):
                # Actual implementation
                pass
    """

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from a prompt.

        Subclasses should integrate caching and rate limiting.
        See class docstring for implementation example.

        Args:
            prompt: The input prompt
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def generate_text_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Generate text from a prompt with streaming.

        Args:
            prompt: The input prompt
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks as they are generated

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If generation fails
        """
        pass


class ImageProvider(Provider):
    """
    Abstract base class for image generation providers.

    Image providers interface with image generation models to create
    images based on text prompts. Supports optional response caching
    and rate limiting.

    Implementation note: Responses are cached as base64-encoded strings
    to allow efficient storage of binary image data.
    """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Generate an image from a text prompt.

        Subclasses can optionally integrate caching and rate limiting
        using the helper methods provided by the Provider base class.

        Args:
            prompt: The text prompt describing the image
            model: Model to use (uses default if not specified)
            width: Image width in pixels
            height: Image height in pixels
            **kwargs: Additional provider-specific parameters

        Returns:
            Image data as bytes (typically PNG format)

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If generation fails
        """
        pass
