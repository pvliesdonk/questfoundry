"""Tests for Provider integration with caching and rate limiting."""

from questfoundry.providers.base import ImageProvider, TextProvider


class MockTextProvider(TextProvider):
    """Mock text provider for testing."""

    def validate_config(self) -> None:
        """Validate configuration."""
        pass

    def generate_text(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Generate text (mock implementation)."""
        return f"Response to: {prompt}"

    def generate_text_streaming(self, prompt: str, model: str | None = None, **kwargs):
        """Generate text streaming (mock implementation)."""
        yield "chunk1"
        yield "chunk2"


class MockImageProvider(ImageProvider):
    """Mock image provider for testing."""

    def validate_config(self) -> None:
        """Validate configuration."""
        pass

    def generate_image(self, prompt: str, model: str | None = None, **kwargs) -> bytes:
        """Generate image (mock implementation)."""
        return b"fake_image_data"


class TestProviderCaching:
    """Tests for Provider cache integration."""

    def test_provider_initializes_cache_by_default(self, tmp_path) -> None:
        """Provider initializes cache by default."""
        provider = MockTextProvider({"cache": {"cache_dir": str(tmp_path / "cache1")}})
        assert provider.cache is not None
        assert provider.cache_config is not None

    def test_provider_cache_disabled(self) -> None:
        """Provider cache can be disabled."""
        provider = MockTextProvider({"cache_enabled": False})
        assert provider.cache is None
        assert provider.cache_config is None

    def test_provider_with_custom_cache_config(self, tmp_path) -> None:
        """Provider accepts custom cache configuration."""
        cache_dir = str(tmp_path / "cache")
        provider = MockTextProvider(
            {
                "cache": {
                    "enabled": True,
                    "cache_dir": cache_dir,
                    "ttl_seconds": 3600,
                }
            }
        )
        assert provider.cache is not None
        assert str(provider.cache_config.cache_dir) == cache_dir
        assert provider.cache_config.ttl_seconds == 3600

    def test_get_cache_key(self, tmp_path) -> None:
        """_get_cache_key generates consistent keys."""
        provider = MockTextProvider({"cache": {"cache_dir": str(tmp_path / "cache2")}})
        key1 = provider._get_cache_key("test prompt", model="gpt-4")
        key2 = provider._get_cache_key("test prompt", model="gpt-4")
        assert key1 == key2

    def test_get_cache_key_different_prompts(self, tmp_path) -> None:
        """_get_cache_key generates different keys for different prompts."""
        provider = MockTextProvider({"cache": {"cache_dir": str(tmp_path / "cache3")}})
        key1 = provider._get_cache_key("prompt1")
        key2 = provider._get_cache_key("prompt2")
        assert key1 != key2

    def test_cache_response(self, tmp_path) -> None:
        """_cache_response stores response."""
        provider = MockTextProvider({"cache": {"cache_dir": str(tmp_path / "cache4")}})
        key = provider._get_cache_key("test")
        provider._cache_response(key, "test response")
        assert provider._get_cached_response(key) == "test response"

    def test_get_cached_response_miss(self, tmp_path) -> None:
        """_get_cached_response returns None for cache miss."""
        provider = MockTextProvider({"cache": {"cache_dir": str(tmp_path / "cache5")}})
        key = provider._get_cache_key("test")
        assert provider._get_cached_response(key) is None

    def test_cache_disabled_get_cached_response(self) -> None:
        """_get_cached_response returns None when cache disabled."""
        provider = MockTextProvider({"cache_enabled": False})
        key = provider._get_cache_key("test")
        assert provider._get_cached_response(key) is None

    def test_cache_disabled_cache_response(self) -> None:
        """_cache_response does nothing when cache disabled."""
        provider = MockTextProvider({"cache_enabled": False})
        key = provider._get_cache_key("test")
        # Should not raise error
        provider._cache_response(key, "response")
        assert provider._get_cached_response(key) is None

    def test_get_cache_stats(self, tmp_path) -> None:
        """get_cache_stats returns cache statistics."""
        provider = MockTextProvider({"cache": {"cache_dir": str(tmp_path / "cache6")}})
        key = provider._get_cache_key("test")
        provider._cache_response(key, "response")

        stats = provider.get_cache_stats()
        assert "entries" in stats
        assert stats["entries"] == 1

    def test_get_cache_stats_disabled(self) -> None:
        """get_cache_stats returns empty dict when disabled."""
        provider = MockTextProvider({"cache_enabled": False})
        stats = provider.get_cache_stats()
        assert stats == {}


class TestProviderRateLimiting:
    """Tests for Provider rate limiting integration."""

    def test_provider_initializes_without_rate_limiter_by_default(self) -> None:
        """Provider doesn't initialize rate limiter by default."""
        provider = MockTextProvider({})
        assert provider.rate_limiter is None
        assert provider.cost_tracker is None

    def test_provider_with_rate_limit_config(self) -> None:
        """Provider initializes rate limiter with config."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 60,
                    "tokens_per_hour": 90000,
                }
            }
        )
        assert provider.rate_limiter is not None
        assert provider.cost_tracker is not None

    def test_check_rate_limit_allowed(self) -> None:
        """_check_rate_limit returns True when allowed."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                }
            }
        )
        assert provider._check_rate_limit(input_tokens=100, output_tokens=100)

    def test_check_rate_limit_denied(self) -> None:
        """_check_rate_limit returns False when limit exceeded."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 1,
                    "tokens_per_hour": 10,  # Very low limit
                }
            }
        )
        # Exceed token limit
        assert not provider._check_rate_limit(input_tokens=100, output_tokens=100)

    def test_check_rate_limit_no_limiter(self) -> None:
        """_check_rate_limit returns True when rate limiter disabled."""
        provider = MockTextProvider({})
        assert provider._check_rate_limit(input_tokens=1000000, output_tokens=1000000)

    def test_record_usage(self) -> None:
        """_record_usage records usage statistics."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                }
            }
        )
        provider._record_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            cost_per_input_1k=0.03,
            cost_per_output_1k=0.06,
        )

        stats = provider.get_rate_limit_stats()
        assert stats["total_requests"] == 1
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 200

    def test_record_usage_cost_tracking(self) -> None:
        """_record_usage tracks costs."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                }
            }
        )
        provider._record_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            cost_per_input_1k=0.03,
            cost_per_output_1k=0.06,
        )

        summary = provider.get_cost_summary()
        assert summary["total_cost"] > 0

    def test_get_rate_limit_stats(self) -> None:
        """get_rate_limit_stats returns rate limit statistics."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                }
            }
        )
        stats = provider.get_rate_limit_stats()
        assert "total_requests" in stats
        assert "available_request_tokens" in stats

    def test_get_rate_limit_stats_disabled(self) -> None:
        """get_rate_limit_stats returns empty dict when disabled."""
        provider = MockTextProvider({})
        stats = provider.get_rate_limit_stats()
        assert stats == {}

    def test_get_cost_summary(self) -> None:
        """get_cost_summary returns cost information."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                }
            }
        )
        provider._record_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            cost_per_input_1k=0.03,
            cost_per_output_1k=0.06,
        )

        summary = provider.get_cost_summary()
        assert "total_cost" in summary
        assert "cost_today" in summary
        assert "by_provider" in summary

    def test_get_cost_summary_disabled(self) -> None:
        """get_cost_summary returns empty dict when disabled."""
        provider = MockTextProvider({})
        summary = provider.get_cost_summary()
        assert summary == {}


class TestProviderIntegration:
    """Integration tests for complete provider functionality."""

    def test_provider_with_both_cache_and_rate_limiting(self) -> None:
        """Provider can use both caching and rate limiting together."""
        provider = MockTextProvider(
            {
                "cache": {"ttl_seconds": 3600},
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                },
            }
        )

        # Should have both
        assert provider.cache is not None
        assert provider.rate_limiter is not None
        assert provider.cost_tracker is not None

    def test_text_provider_example_pattern(self, tmp_path) -> None:
        """Test the example pattern from TextProvider docstring."""
        provider = MockTextProvider(
            {
                "cache": {
                    "ttl_seconds": 3600,
                    "cache_dir": str(tmp_path / "cache_example"),
                },
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                },
            }
        )

        prompt = "test prompt"

        # 1. Generate cache key
        cache_key = provider._get_cache_key(prompt)

        # 2. Check cache first (miss)
        cached = provider._get_cached_response(cache_key)
        assert cached is None

        # 3. Check rate limits
        assert provider._check_rate_limit(input_tokens=100)

        # 4. Make actual API call
        response = "Response to: test prompt"

        # 5. Cache response
        provider._cache_response(cache_key, response)

        # 6. Record usage
        provider._record_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=250,
            cost_per_input_1k=0.03,
            cost_per_output_1k=0.06,
        )

        # Now cache should have response
        cached = provider._get_cached_response(cache_key)
        assert cached == response

        # Stats should be recorded
        rate_stats = provider.get_rate_limit_stats()
        assert rate_stats["total_requests"] == 1

    def test_image_provider_basic(self) -> None:
        """ImageProvider also supports caching and rate limiting."""
        provider = MockImageProvider(
            {
                "cache": {"ttl_seconds": 3600},
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                },
            }
        )

        assert provider.cache is not None
        assert provider.rate_limiter is not None

    def test_provider_close(self) -> None:
        """Provider.close() can be called safely."""
        provider = MockTextProvider({})
        # Should not raise
        provider.close()

    def test_provider_context_manager(self) -> None:
        """Provider works as context manager."""
        with MockTextProvider({}) as provider:
            assert provider is not None
            assert provider.cache is not None


class TestProviderEdgeCases:
    """Edge case tests for Provider integration."""

    def test_cache_response_with_custom_ttl(self) -> None:
        """_cache_response respects custom TTL."""
        provider = MockTextProvider({})
        key = provider._get_cache_key("test")

        # Cache with custom TTL
        provider._cache_response(key, "response", ttl=10)

        # Should be there immediately
        assert provider._get_cached_response(key) == "response"

    def test_record_usage_without_rate_limiter(self) -> None:
        """_record_usage handles missing rate limiter gracefully."""
        provider = MockTextProvider({})
        # Should not raise
        provider._record_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
        )

    def test_multiple_providers_independent_caches(self, tmp_path) -> None:
        """Multiple providers have independent caches."""
        cache_dir1 = str(tmp_path / "cache_p1")
        cache_dir2 = str(tmp_path / "cache_p2")
        provider1 = MockTextProvider({"cache": {"cache_dir": cache_dir1}})
        provider2 = MockTextProvider({"cache": {"cache_dir": cache_dir2}})

        key = "test_key"
        provider1._cache_response(key, "response1")
        provider2._cache_response(key, "response2")

        # Each should have its own cache
        assert provider1._get_cached_response(key) == "response1"
        assert provider2._get_cached_response(key) == "response2"

    def test_rate_limiter_stats_before_usage(self) -> None:
        """Rate limiter stats accessible before any usage."""
        provider = MockTextProvider(
            {
                "rate_limit": {
                    "requests_per_minute": 10,
                    "tokens_per_hour": 10000,
                }
            }
        )
        stats = provider.get_rate_limit_stats()
        assert stats["total_requests"] == 0
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0
