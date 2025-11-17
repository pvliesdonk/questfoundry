"""FastAPI dependencies for connection pooling and storage backends

This module provides shared database connection pools and Redis clients
that are initialized once at application startup and reused across requests.
"""

from __future__ import annotations

from typing import Literal, cast

import redis
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import Settings
from .storage import PostgresStore, ValkeyStore

# Global pool instances (initialized at startup)
_postgres_pool: ConnectionPool | None = None
_redis_client: redis.Redis | None = None


def init_pools(settings: Settings) -> None:
    """Initialize connection pools at application startup.

    This should be called from the FastAPI startup event handler.

    Args:
        settings: Application settings containing database URLs
    """
    global _postgres_pool, _redis_client

    # Initialize PostgreSQL connection pool
    _postgres_pool = ConnectionPool(
        settings.postgres_url,
        min_size=2,
        max_size=10,
        kwargs={"row_factory": dict_row},
    )

    # Initialize Redis client (thread-safe, can be shared)
    _redis_client = cast(
        redis.Redis,
        redis.from_url(settings.redis_url, decode_responses=True),  # type: ignore[no-untyped-call]
    )


def close_pools() -> None:
    """Close connection pools at application shutdown.

    This should be called from the FastAPI shutdown event handler.
    """
    global _postgres_pool, _redis_client

    if _postgres_pool is not None:
        _postgres_pool.close()
        _postgres_pool = None

    if _redis_client is not None:
        _redis_client.close()
        _redis_client = None


def get_postgres_pool() -> ConnectionPool:
    """Get the shared PostgreSQL connection pool.

    Returns:
        Shared ConnectionPool instance

    Raises:
        RuntimeError: If pool not initialized (startup event not run)
    """
    if _postgres_pool is None:
        raise RuntimeError(
            "PostgreSQL pool not initialized. "
            "Ensure init_pools() is called at application startup."
        )
    return _postgres_pool


def get_redis_client() -> redis.Redis:
    """Get the shared Redis client.

    Returns:
        Shared Redis client instance

    Raises:
        RuntimeError: If client not initialized (startup event not run)
    """
    if _redis_client is None:
        raise RuntimeError(
            "Redis client not initialized. "
            "Ensure init_pools() is called at application startup."
        )
    return _redis_client


def create_storage_backend(
    project_id: str,
    storage: Literal["hot", "cold"],
    postgres_pool: ConnectionPool,
    redis_client: redis.Redis,
) -> PostgresStore | ValkeyStore:
    """Create a storage backend instance using shared connection pools.

    This function creates storage backend instances that use the shared
    connection pools, preventing resource exhaustion. It should be called
    with the pools obtained from FastAPI dependency injection.

    Args:
        project_id: Project identifier for scoping
        storage: Storage type ("hot" for Redis, "cold" for PostgreSQL)
        postgres_pool: Shared PostgreSQL connection pool
        redis_client: Shared Redis client

    Returns:
        Configured storage backend instance

    Example:
        postgres_pool = Depends(get_postgres_pool)
        redis_client = Depends(get_redis_client)
        backend = create_storage_backend(
            project_id, "cold", postgres_pool, redis_client
        )
    """
    if storage == "cold":
        cold_store = PostgresStore.__new__(PostgresStore)
        cold_store.project_id = project_id
        cold_store.pool = postgres_pool
        return cold_store
    if storage == "hot":
        hot_store = ValkeyStore.__new__(ValkeyStore)
        hot_store.project_id = project_id
        hot_store.client = redis_client
        hot_store.ttl_seconds = 86400  # 24 hours
        return hot_store
    raise ValueError(f"Unknown storage type: {storage}")
