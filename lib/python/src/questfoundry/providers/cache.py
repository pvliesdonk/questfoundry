"""Response caching system for QuestFoundry providers."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def generate_cache_key(
    provider: str,
    model: str,
    prompt: str,
    **kwargs: Any,
) -> str:
    """Generate a cache key from request parameters.

    Creates a deterministic SHA256 hash-based cache key from the provider,
    model, prompt, and other relevant parameters. This ensures:
    - Same inputs always produce same key
    - Keys are reasonably sized (hashes, not full text)
    - All parameter combinations are unique

    Args:
        provider: Provider name (e.g., 'openai', 'ollama')
        model: Model name (e.g., 'gpt-4o', 'llama3')
        prompt: The input prompt text
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        Cache key string (e.g., 'cache:a1b2c3...')
    """
    key_data = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 2000),
        "top_p": kwargs.get("top_p"),
        "frequency_penalty": kwargs.get("frequency_penalty"),
        "presence_penalty": kwargs.get("presence_penalty"),
    }

    # Remove None values to keep keys clean
    key_data = {k: v for k, v in key_data.items() if v is not None}

    # Create deterministic JSON string
    key_str = json.dumps(key_data, sort_keys=True)

    # Hash it
    hash_hex = hashlib.sha256(key_str.encode()).hexdigest()

    return f"cache:{hash_hex}"


@dataclass
class CacheConfig:
    """Configuration for response caching.

    Controls caching behavior including storage backend, TTL,
    size limits, and per-provider overrides.

    Attributes:
        enabled: Whether caching is enabled globally (default: False for opt-in)
        backend: Storage backend ('file' or 'memory')
        cache_dir: Directory for file-based cache
        ttl_seconds: Default TTL for cached responses (seconds)
        max_cache_size_mb: Maximum total cache size (megabytes)
        cleanup_interval_seconds: How often to clean expired entries
        per_provider: Per-provider override configurations
    """

    enabled: bool = False
    backend: str = "file"  # 'file', 'memory', or 'redis' (future)
    cache_dir: Optional[Path] = None
    ttl_seconds: int = 86400  # 24 hours
    max_cache_size_mb: int = 500
    cleanup_interval_seconds: int = 3600  # 1 hour
    per_provider: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default cache directory if not provided."""
        if self.cache_dir is None:
            self.cache_dir = Path.cwd() / ".questfoundry" / "cache"
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheConfig":
        """Create CacheConfig from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            CacheConfig instance
        """
        config_data = data.copy()
        if "cache_dir" in config_data and isinstance(config_data["cache_dir"], str):
            config_data["cache_dir"] = Path(config_data["cache_dir"])
        return cls(**config_data)

    def get_ttl_for_provider(self, provider: str) -> int:
        """Get TTL for a specific provider.

        Args:
            provider: Provider name

        Returns:
            TTL in seconds for this provider
        """
        provider_config = self.per_provider.get(provider, {})
        return provider_config.get("ttl_seconds", self.ttl_seconds)

    def is_enabled_for_provider(self, provider: str) -> bool:
        """Check if caching is enabled for a provider.

        Args:
            provider: Provider name

        Returns:
            True if caching enabled for this provider
        """
        if not self.enabled:
            return False
        provider_config = self.per_provider.get(provider, {})
        return provider_config.get("enabled", True)


class ResponseCache:
    """File-based response cache with TTL support.

    Caches provider responses to avoid duplicate API calls.
    Uses file-based storage with automatic expiration.

    The cache stores responses in a directory structure:
    ```
    .questfoundry/cache/
    ├── ab/
    │   ├── abcd1234567890.json          # Cached response
    │   └── abcd1234567890.meta.json     # Metadata (TTL, timestamp)
    ├── cd/
    ...
    ```

    Example:
        ```python
        cache = ResponseCache(cache_dir=Path(".questfoundry/cache"))
        cache.set("my_key", "response_data", ttl=86400)
        result = cache.get("my_key")
        if result:
            print(f"Cache hit: {result}")
        ```
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = 86400,
    ) -> None:
        """Initialize the response cache.

        Args:
            cache_dir: Directory for cache storage
            ttl_seconds: Default TTL for cached responses
        """
        logger.debug("Initializing ResponseCache with TTL=%d seconds", ttl_seconds)
        self.cache_dir = cache_dir or Path.cwd() / ".questfoundry" / "cache"
        self.ttl_seconds = ttl_seconds
        logger.trace("Cache directory: %s", self.cache_dir)

        # Create cache directory if it doesn't exist
        logger.trace("Creating cache directory if needed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.trace("ResponseCache initialized successfully")

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key.

        Uses first 2 characters of key as directory prefix
        to avoid having all cache files in one directory.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        prefix = key[:2]
        return self.cache_dir / prefix / f"{key}.json"

    def _get_metadata_path(self, key: str) -> Path:
        """Get file path for cache metadata.

        Args:
            key: Cache key

        Returns:
            Path to metadata file
        """
        prefix = key[:2]
        return self.cache_dir / prefix / f"{key}.meta.json"

    def get(self, key: str) -> Optional[str]:
        """Get cached response if not expired.

        Checks if cache entry exists and is still valid (not expired).
        Automatically removes expired entries.

        Args:
            key: Cache key

        Returns:
            Cached response string, or None if not found or expired
        """
        logger.trace("Looking up cache key: %s", key)
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        if not cache_path.exists():
            logger.trace("Cache key not found: %s", key)
            return None

        try:
            # Check if expired
            logger.trace("Checking cache expiration for key: %s", key)
            metadata = json.loads(meta_path.read_text())
            if time.time() > metadata["expires_at"]:
                # Clean up expired entries
                logger.debug("Cache entry expired, cleaning up key: %s", key)
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None

            # Cache hit
            logger.debug("Cache hit for key: %s", key)
            return cache_path.read_text()
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            # Corrupted cache, clean up
            logger.warning("Corrupted cache entry for key %s: %s", key, str(e))
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> None:
        """Store response with metadata.

        Args:
            key: Cache key
            value: Response value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        logger.trace("Caching value for key: %s", key)
        ttl = ttl or self.ttl_seconds
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        # Create directory if needed
        logger.trace("Creating cache directory if needed for key: %s", key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write cache file (atomic: write to temp, then rename)
            logger.trace("Writing cache file for key: %s", key)
            temp_file = cache_path.with_suffix(".tmp")
            temp_file.write_text(value)
            temp_file.replace(cache_path)

            # Write metadata
            logger.trace(
                "Writing cache metadata for key: %s (TTL=%d seconds)", key, ttl
            )
            now = time.time()
            metadata = {
                "timestamp": now,
                "ttl_seconds": ttl,
                "expires_at": now + ttl,
            }
            meta_path.write_text(json.dumps(metadata))
            logger.debug("Response cached successfully with key: %s", key)
        except Exception as e:
            # Cache failures are logged but not raised to allow graceful
            # degradation. The application can continue without caching.
            logger.warning(
                "Error caching response for key %s: %s", key, str(e), exc_info=True
            )

    def clear(self) -> None:
        """Clear all cached entries.

        Removes the entire cache directory and recreates it empty.
        """
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Scans all metadata files and removes entries that have expired.

        Returns:
            Number of entries removed
        """
        logger.debug("Starting cleanup of expired cache entries")

        if not self.cache_dir.exists():
            logger.trace("Cache directory does not exist, nothing to clean")
            return 0

        removed_count = 0
        current_time = time.time()

        for meta_file in self.cache_dir.rglob("*.meta.json"):
            try:
                metadata = json.loads(meta_file.read_text())
                if current_time > metadata["expires_at"]:
                    # Remove both meta and cache file
                    logger.trace("Removing expired cache entry: %s", meta_file.name)
                    cache_file = meta_file.with_suffix("")
                    meta_file.unlink(missing_ok=True)
                    cache_file.unlink(missing_ok=True)
                    removed_count += 1
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                # Clean up corrupted files
                logger.warning("Cleaning up corrupted cache file: %s", meta_file.name)
                meta_file.unlink(missing_ok=True)
                removed_count += 1

        logger.info(
            "Cache cleanup completed, removed %d expired entries", removed_count
        )
        return removed_count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with 'entries' and 'total_size_mb' keys
        """
        if not self.cache_dir.exists():
            return {"entries": 0, "total_size_mb": 0.0}

        total_size = 0
        count = 0

        for cache_file in self.cache_dir.rglob("*.json"):
            if ".meta.json" not in str(cache_file):
                count += 1
                total_size += cache_file.stat().st_size

        return {
            "entries": count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
