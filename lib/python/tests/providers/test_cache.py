"""Tests for response caching system."""

import json
import time
from pathlib import Path

import pytest

from questfoundry.providers.cache import CacheConfig, ResponseCache, generate_cache_key


class TestGenerateCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_generation_consistent(self) -> None:
        """Same inputs produce same cache key."""
        key1 = generate_cache_key("openai", "gpt-4", "hello")
        key2 = generate_cache_key("openai", "gpt-4", "hello")
        assert key1 == key2

    def test_cache_key_generation_different_prompts(self) -> None:
        """Different prompts produce different keys."""
        key1 = generate_cache_key("openai", "gpt-4", "hello")
        key2 = generate_cache_key("openai", "gpt-4", "world")
        assert key1 != key2

    def test_cache_key_includes_model(self) -> None:
        """Cache key includes model name."""
        key1 = generate_cache_key("openai", "gpt-4", "hello")
        key2 = generate_cache_key("openai", "gpt-3.5-turbo", "hello")
        assert key1 != key2

    def test_cache_key_includes_provider(self) -> None:
        """Cache key includes provider name."""
        key1 = generate_cache_key("openai", "gpt-4", "hello")
        key2 = generate_cache_key("ollama", "gpt-4", "hello")
        assert key1 != key2

    def test_cache_key_includes_temperature(self) -> None:
        """Cache key includes temperature."""
        key1 = generate_cache_key("openai", "gpt-4", "hello", temperature=0.7)
        key2 = generate_cache_key("openai", "gpt-4", "hello", temperature=0.9)
        assert key1 != key2

    def test_cache_key_includes_max_tokens(self) -> None:
        """Cache key includes max_tokens."""
        key1 = generate_cache_key("openai", "gpt-4", "hello", max_tokens=100)
        key2 = generate_cache_key("openai", "gpt-4", "hello", max_tokens=200)
        assert key1 != key2

    def test_cache_key_format(self) -> None:
        """Cache key has correct format."""
        key = generate_cache_key("openai", "gpt-4", "hello")
        assert key.startswith("cache:")
        assert len(key) > 10  # Should be reasonably long (SHA256 hash)

    def test_cache_key_ignores_none_values(self) -> None:
        """Cache key ignores None parameter values."""
        key1 = generate_cache_key("openai", "gpt-4", "hello", top_p=None)
        key2 = generate_cache_key("openai", "gpt-4", "hello")
        assert key1 == key2


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_cache_config_defaults(self) -> None:
        """CacheConfig has reasonable defaults."""
        config = CacheConfig()
        assert config.enabled is False  # Opt-in by default
        assert config.backend == "file"
        assert config.ttl_seconds == 86400  # 24 hours
        assert config.max_cache_size_mb == 500

    def test_cache_config_custom_values(self) -> None:
        """CacheConfig can be customized."""
        config = CacheConfig(
            enabled=False,
            ttl_seconds=3600,
            max_cache_size_mb=100,
        )
        assert config.enabled is False
        assert config.ttl_seconds == 3600
        assert config.max_cache_size_mb == 100

    def test_cache_config_default_cache_dir(self) -> None:
        """Default cache directory is set."""
        config = CacheConfig()
        assert config.cache_dir is not None
        assert config.cache_dir == Path.cwd() / ".questfoundry" / "cache"

    def test_cache_config_custom_cache_dir_string(self) -> None:
        """Cache dir can be provided as string."""
        config = CacheConfig(cache_dir="/tmp/cache")
        assert config.cache_dir == Path("/tmp/cache")

    def test_cache_config_custom_cache_dir_path(self) -> None:
        """Cache dir can be provided as Path."""
        cache_path = Path("/tmp/cache")
        config = CacheConfig(cache_dir=cache_path)
        assert config.cache_dir == cache_path

    def test_cache_config_from_dict(self) -> None:
        """CacheConfig can be created from dictionary."""
        config_dict = {
            "enabled": True,
            "backend": "file",
            "cache_dir": "/tmp/cache",
            "ttl_seconds": 86400,
        }
        config = CacheConfig.from_dict(config_dict)
        assert config.enabled is True
        assert config.cache_dir == Path("/tmp/cache")

    def test_cache_config_get_ttl_for_provider_default(self) -> None:
        """Default TTL returned for unknown provider."""
        config = CacheConfig(ttl_seconds=86400)
        assert config.get_ttl_for_provider("openai") == 86400

    def test_cache_config_get_ttl_for_provider_override(self) -> None:
        """Per-provider TTL overrides default."""
        config = CacheConfig(
            ttl_seconds=86400,
            per_provider={"openai": {"ttl_seconds": 172800}},
        )
        assert config.get_ttl_for_provider("openai") == 172800
        assert config.get_ttl_for_provider("gemini") == 86400

    def test_cache_config_is_enabled_for_provider_global_disabled(self) -> None:
        """Provider caching disabled if global is disabled."""
        config = CacheConfig(enabled=False)
        assert config.is_enabled_for_provider("openai") is False

    def test_cache_config_is_enabled_for_provider_default(self) -> None:
        """Provider caching enabled by default."""
        config = CacheConfig(enabled=True)
        assert config.is_enabled_for_provider("openai") is True

    def test_cache_config_is_enabled_for_provider_override(self) -> None:
        """Per-provider can disable caching."""
        config = CacheConfig(
            enabled=True,
            per_provider={"ollama": {"enabled": False}},
        )
        assert config.is_enabled_for_provider("openai") is True
        assert config.is_enabled_for_provider("ollama") is False


class TestResponseCache:
    """Tests for ResponseCache."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> ResponseCache:
        """Create a test cache with temp directory."""
        return ResponseCache(cache_dir=tmp_path / "cache")

    def test_cache_set_get(self, cache: ResponseCache) -> None:
        """Basic set and get operations."""
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

    def test_cache_miss(self, cache: ResponseCache) -> None:
        """Get non-existent key returns None."""
        assert cache.get("nonexistent") is None

    def test_cache_overwrite(self, cache: ResponseCache) -> None:
        """Overwriting key updates value."""
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_cache_expiration(self, cache: ResponseCache) -> None:
        """Cached items expire after TTL."""
        cache.set("key", "value", ttl=1)
        assert cache.get("key") == "value"
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_cache_expiration_custom_ttl(self) -> None:
        """Custom TTL per item works."""
        cache = ResponseCache(ttl_seconds=10)
        cache.set("key", "value", ttl=1)
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_cache_directory_structure(self, cache: ResponseCache) -> None:
        """Cache creates proper directory structure."""
        cache.set("abcdef123", "test")
        # Should create ab/abcdef123.json
        path = cache._get_cache_path("abcdef123")
        assert path.exists()
        assert path.parent.name == "ab"

    def test_cache_cleanup_expired(self, cache: ResponseCache) -> None:
        """cleanup_expired removes stale entries."""
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=1)
        time.sleep(1.1)

        removed = cache.cleanup_expired()
        assert removed == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_cleanup_preserves_valid(self, cache: ResponseCache) -> None:
        """cleanup_expired preserves valid entries."""
        cache.set("old", "value1", ttl=1)
        cache.set("new", "value2", ttl=10000)
        time.sleep(1.1)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("old") is None
        assert cache.get("new") == "value2"

    def test_cache_stats(self, cache: ResponseCache) -> None:
        """Cache statistics are correct."""
        cache.set("key1", "x" * 10000)  # 10KB
        cache.set("key2", "y" * 10000)  # 10KB

        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["total_size_mb"] > 0

    def test_cache_stats_empty(self, cache: ResponseCache) -> None:
        """Stats for empty cache are zero."""
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["total_size_mb"] == 0.0

    def test_cache_clear(self, cache: ResponseCache) -> None:
        """Clear removes all entries."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_large_value(self, cache: ResponseCache) -> None:
        """Cache handles large responses."""
        large_value = "x" * 100000  # 100KB
        cache.set("large", large_value)
        assert cache.get("large") == large_value

    def test_cache_json_value(self, cache: ResponseCache) -> None:
        """Cache handles JSON values."""
        json_value = json.dumps({"key": "value", "nested": {"data": 123}})
        cache.set("json", json_value)
        assert cache.get("json") == json_value

    def test_cache_special_characters(self, cache: ResponseCache) -> None:
        """Cache handles special characters."""
        special_value = 'Hello 世界 🌍 \n\t"quoted"'
        cache.set("special", special_value)
        assert cache.get("special") == special_value

    def test_cache_multiple_keys(self, cache: ResponseCache) -> None:
        """Cache handles multiple independent keys."""
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

        for i in range(10):
            assert cache.get(f"key_{i}") == f"value_{i}"

    def test_cache_corrupted_metadata(self, cache: ResponseCache) -> None:
        """Cache handles corrupted metadata gracefully."""
        cache.set("key", "value")

        # Corrupt the metadata
        meta_path = cache._get_metadata_path("key")
        meta_path.write_text("invalid json {{{")

        # Should return None and clean up
        result = cache.get("key")
        assert result is None
        assert not meta_path.exists()

    def test_cache_metadata_structure(self, cache: ResponseCache) -> None:
        """Cached metadata has correct structure."""
        before = time.time()
        cache.set("key", "value", ttl=3600)
        after = time.time()

        meta_path = cache._get_metadata_path("key")
        metadata = json.loads(meta_path.read_text())

        assert "timestamp" in metadata
        assert "ttl_seconds" in metadata
        assert "expires_at" in metadata
        assert before <= metadata["timestamp"] <= after
        assert metadata["ttl_seconds"] == 3600
        assert metadata["expires_at"] == metadata["timestamp"] + 3600

    def test_cache_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Cache initialization creates directory."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        ResponseCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_cache_concurrent_writes(self, cache: ResponseCache) -> None:
        """Cache handles concurrent operations reasonably."""
        import threading

        results = []

        def worker(thread_id: int) -> None:
            for i in range(10):
                # Use unique keys per thread to avoid race conditions
                key = f"key_thread{thread_id}_{i}"
                cache.set(key, f"value_{i}")
                result = cache.get(key)
                if result is not None:
                    results.append(result)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should successfully write and read 10 items
        assert len(results) == 30
        assert all(r is not None for r in results)

    def test_cache_get_stats_after_operations(self, cache: ResponseCache) -> None:
        """Stats updated correctly after cache operations."""
        cache.set("key1", "value1")
        stats1 = cache.get_stats()
        assert stats1["entries"] == 1

        cache.set("key2", "value2")
        stats2 = cache.get_stats()
        assert stats2["entries"] == 2

        cache.clear()
        stats3 = cache.get_stats()
        assert stats3["entries"] == 0
