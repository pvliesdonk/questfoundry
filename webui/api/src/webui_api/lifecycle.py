"""Core request lifecycle for orchestrator instantiation

This module implements the mandated request lifecycle pattern for the WebUI API.
Every request that interacts with QuestFoundry follows this pattern:

1. Acquire distributed lock for the project
2. Instantiate storage backends (project-scoped)
3. Instantiate library components (user-scoped provider config)
4. Yield orchestrator for use by endpoint handler
5. Clean up connections
6. Release lock (automatic via context manager)

All library objects are created fresh per request and discarded after,
ensuring complete isolation between users and requests.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import redis
from questfoundry.orchestrator import Orchestrator
from questfoundry.providers.config import ProviderConfig
from questfoundry.providers.registry import ProviderRegistry
from questfoundry.roles.registry import RoleRegistry
from questfoundry.state.workspace import WorkspaceManager

from .config import settings
from .locking import ProjectLock
from .storage import PostgresStore, ValkeyStore

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def orchestrator_context(
    project_id: str, user_id: str, user_provider_config: ProviderConfig
) -> Generator[Orchestrator, None, None]:
    """
    Context manager for orchestrator lifecycle.

    This implements the complete request lifecycle pattern:
    - Acquires distributed lock to prevent concurrent writes
    - Instantiates project-scoped storage backends
    - Instantiates user-scoped library components
    - Creates orchestrator with all dependencies
    - Yields orchestrator for use
    - Automatically cleans up and releases lock

    All objects created here are discarded after the context exits.
    This ensures complete isolation between requests and prevents
    state leakage between users.

    Args:
        project_id: Project identifier for scoping storage
        user_id: User identifier for lock ownership
        user_provider_config: User's decrypted BYOK provider config

    Yields:
        Orchestrator instance ready for use

    Raises:
        HTTPException: 423 if project is locked by another user

    Example:
        >>> with orchestrator_context(project_id, user_id, config) as orch:
        ...     result = orch.execute_goal("Create a new hook")
    """
    # Create Redis client for locking
    redis_client = redis.from_url(settings.redis_url, decode_responses=False)

    try:
        # Create lock manager
        lock = ProjectLock(redis_client, settings.lock_timeout)

        with lock.acquire(project_id, user_id):
            # Instantiate storage backends (project-scoped)
            cold_store = PostgresStore(settings.postgres_url, project_id)
            hot_store = ValkeyStore(settings.redis_url, project_id, ttl_seconds=86400)

            try:
                # Instantiate library components
                provider_reg = ProviderRegistry(config=user_provider_config)

                # Determine spec path
                if settings.spec_path:
                    spec_path = Path(settings.spec_path)
                else:
                    # Use bundled spec from questfoundry-py
                    # The library will handle this automatically
                    spec_path = None  # type: ignore

                role_reg = RoleRegistry(provider_reg, spec_path=spec_path)
                workspace = WorkspaceManager(cold=cold_store, hot=hot_store)

                # Create orchestrator
                orchestrator = Orchestrator(
                    workspace=workspace,
                    provider_registry=provider_reg,
                    role_registry=role_reg,
                    spec_path=spec_path,
                )

                yield orchestrator

            finally:
                # Cleanup storage connections
                cold_store.close()
                hot_store.close()

    finally:
        # Cleanup Redis client
        redis_client.close()
