"""Project locking mechanism for concurrent write protection

This module provides a Redis-based distributed locking mechanism to prevent
concurrent writes to the same project from different users or requests.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import redis
from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Generator


class ProjectLock:
    """
    Redis-based distributed lock for project access.

    Prevents concurrent writes to the same project by different users.
    Uses Redis SET with NX (not exists) and EX (expiration) for atomic
    lock acquisition.

    Thread Safety:
        Redis client is thread-safe. Multiple ProjectLock instances can
        safely share the same Redis client.

    Example:
        >>> lock = ProjectLock(redis_client, timeout=300)
        >>> with lock.acquire("project-123", "user-456"):
        ...     # Do work on project
        ...     pass
    """

    def __init__(self, redis_client: redis.Redis, timeout: int = 300):
        """
        Initialize project lock.

        Args:
            redis_client: Redis client instance
            timeout: Lock timeout in seconds (default 5 minutes)
        """
        self.client = redis_client
        self.timeout = timeout

    @contextmanager
    def acquire(self, project_id: str, user_id: str) -> Generator[None, None, None]:
        """
        Acquire lock for project.

        Attempts to acquire an exclusive lock on the project. If the lock
        is already held by another user, raises HTTPException 423 (Locked).

        If the same user already holds the lock, allows re-entry.

        The lock is automatically released when the context exits, or
        expires after the timeout period.

        Args:
            project_id: Project to lock
            user_id: User requesting lock

        Yields:
            None (context manager)

        Raises:
            HTTPException: 423 if project is locked by another user
        """
        lock_key = f"lock:project:{project_id}"

        # Try to acquire lock atomically
        acquired = self.client.set(
            lock_key,
            user_id,
            nx=True,  # Only set if not exists
            ex=self.timeout,  # Expire after timeout
        )

        if not acquired:
            # Lock exists, check who owns it
            owner = self.client.get(lock_key)
            if owner and owner.decode("utf-8") == user_id:
                # Same user already has lock, allow re-entry
                pass
            else:
                raise HTTPException(
                    status_code=423,
                    detail=f"Project {project_id} is locked by another user",
                )

        try:
            yield
        finally:
            # Release lock only if we still own it
            current_owner = self.client.get(lock_key)
            if current_owner and current_owner.decode("utf-8") == user_id:
                self.client.delete(lock_key)
