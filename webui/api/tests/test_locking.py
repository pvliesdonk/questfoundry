"""Unit tests for project locking mechanism"""

import os
import time

import pytest
import redis
from fastapi import HTTPException

from webui_api.locking import ProjectLock

# Skip all tests if Redis is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_REDIS_URL"),
    reason="Redis not available. Set TEST_REDIS_URL to run these tests.",
)


@pytest.fixture
def redis_client():
    """Create Redis client for testing"""
    url = os.getenv("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL not set")

    client = redis.from_url(url, decode_responses=False)

    yield client

    # Cleanup all test locks
    for key in client.scan_iter(match="lock:project:*", count=100):
        client.delete(key)

    client.close()


@pytest.fixture
def lock(redis_client):
    """Create ProjectLock instance"""
    return ProjectLock(redis_client, timeout=60)


class TestProjectLock:
    """Test project locking mechanism"""

    def test_acquire_new_lock(self, lock, redis_client):
        """Test acquiring a new lock"""
        with lock.acquire("test-project", "user1"):
            # Lock should exist
            value = redis_client.get("lock:project:test-project")
            assert value is not None
            assert value.decode("utf-8") == "user1"

        # Lock should be released after context
        value = redis_client.get("lock:project:test-project")
        assert value is None

    def test_concurrent_lock_blocked(self, lock):
        """Test that concurrent lock acquisition is blocked"""
        with lock.acquire("test-project", "user1"):
            # Try to acquire same lock with different user
            with pytest.raises(HTTPException) as exc_info:
                with lock.acquire("test-project", "user2"):
                    pass

            assert exc_info.value.status_code == 423
            assert "locked by another user" in exc_info.value.detail

    def test_same_user_reacquire(self, lock):
        """Test that same user can re-acquire their own lock"""
        with lock.acquire("test-project", "user1"):
            # Same user should be able to re-acquire
            with lock.acquire("test-project", "user1"):
                # Should not raise
                pass

    def test_lock_timeout(self, redis_client):
        """Test that lock expires after timeout"""
        short_lock = ProjectLock(redis_client, timeout=2)

        with short_lock.acquire("test-project", "user1"):
            pass

        # Wait for expiration
        time.sleep(3)

        # Lock should be expired
        value = redis_client.get("lock:project:test-project")
        assert value is None

    def test_different_projects_independent(self, lock):
        """Test that locks on different projects are independent"""
        with lock.acquire("project1", "user1"):
            # Should be able to lock different project
            with lock.acquire("project2", "user1"):
                # Both locks held
                pass

    def test_lock_released_on_exception(self, lock, redis_client):
        """Test that lock is released even if exception occurs"""
        try:
            with lock.acquire("test-project", "user1"):
                # Verify lock exists
                value = redis_client.get("lock:project:test-project")
                assert value is not None

                # Raise exception
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released
        value = redis_client.get("lock:project:test-project")
        assert value is None

    def test_lock_not_released_if_stolen(self, lock, redis_client):
        """Test that lock is not deleted if ownership changed"""
        with lock.acquire("test-project", "user1"):
            # Simulate lock being stolen by manually changing owner
            redis_client.set("lock:project:test-project", "user2", ex=60)

            # When context exits, should not delete (not owner)
            pass

        # Lock should still exist (owned by user2)
        value = redis_client.get("lock:project:test-project")
        assert value is not None
        assert value.decode("utf-8") == "user2"

        # Cleanup
        redis_client.delete("lock:project:test-project")
