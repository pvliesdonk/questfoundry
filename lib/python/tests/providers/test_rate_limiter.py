"""Tests for rate limiting and cost tracking."""

from datetime import datetime

import pytest

from questfoundry.providers.rate_limiter import (
    CostTracker,
    RateLimitConfig,
    RateLimiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_config_defaults(self) -> None:
        """RateLimitConfig has reasonable defaults."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.tokens_per_hour == 1000
        assert config.cost_per_day is None
        assert config.cost_per_input_token == 0.0
        assert config.cost_per_output_token == 0.0

    def test_config_custom_values(self) -> None:
        """RateLimitConfig can be customized."""
        config = RateLimitConfig(
            requests_per_minute=90,
            tokens_per_hour=90000,
            cost_per_day=100.0,
            cost_per_input_token=0.03,
            cost_per_output_token=0.06,
        )
        assert config.requests_per_minute == 90
        assert config.tokens_per_hour == 90000
        assert config.cost_per_day == 100.0
        assert config.cost_per_input_token == 0.03
        assert config.cost_per_output_token == 0.06


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_basic_acquire(self) -> None:
        """Basic token acquisition."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)
        assert limiter.acquire(num_requests=1) is True

    def test_exhaustion(self) -> None:
        """Requests denied when tokens exhausted."""
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)

        assert limiter.acquire(num_requests=1) is True
        assert limiter.acquire(num_requests=1) is True
        assert limiter.acquire(num_requests=1) is False

    def test_token_refill(self) -> None:
        """Tokens refill over time."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        # Exhaust tokens
        for _ in range(10):
            assert limiter.acquire(num_requests=1)

        # Should be empty now
        assert not limiter.acquire(num_requests=1)

        # Simulate 60 seconds passing
        limiter.last_request_refill -= 60

        # Should have tokens again
        assert limiter.acquire(num_requests=1)

    def test_token_bucket_limit(self) -> None:
        """Token bucket limits work."""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_hour=1000,
        )
        limiter = RateLimiter(config)

        # Should deny if tokens exceed hourly limit
        assert not limiter.acquire(input_tokens=600, output_tokens=500)

        # Should allow if within limit
        assert limiter.acquire(input_tokens=400, output_tokens=500)

    def test_cost_limit(self) -> None:
        """Cost limits work."""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_hour=10000,
            cost_per_day=10.0,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
        )
        limiter = RateLimiter(config)

        # $10/day = 1000 cents
        # 1000 input tokens = 1 cent
        # 500 output tokens = 1 cent
        # Total = 2 cents (under limit)
        assert limiter.acquire(input_tokens=1000, output_tokens=500)

        # 10000 more would be 20 cents total, still under 10000 cents
        assert limiter.acquire(input_tokens=1000, output_tokens=500)

    def test_check_limit_non_consuming(self) -> None:
        """check_limit doesn't consume tokens."""
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)

        # Check limit shouldn't consume
        assert limiter.check_limit(num_requests=1) is True
        assert limiter.check_limit(num_requests=1) is True
        assert limiter.check_limit(num_requests=1) is True

        # But acquire should still work
        assert limiter.acquire(num_requests=1) is True
        assert limiter.acquire(num_requests=1) is True
        assert limiter.acquire(num_requests=1) is False

    def test_usage_tracking(self) -> None:
        """Usage is tracked correctly."""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        limiter.record_usage(100, 200, 1)
        limiter.record_usage(50, 150, 1)

        stats = limiter.get_stats()
        assert stats["total_requests"] == 2
        assert stats["total_input_tokens"] == 150
        assert stats["total_output_tokens"] == 350

    def test_get_stats(self) -> None:
        """Stats include all information."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        limiter.acquire(num_requests=1)
        limiter.record_usage(100, 200)

        stats = limiter.get_stats()
        assert "total_requests" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_cost" in stats
        assert "available_request_tokens" in stats
        assert "available_token_tokens" in stats

    def test_wait_until_available_immediate(self) -> None:
        """wait_until_available returns immediately if available."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        result = limiter.wait_until_available(num_requests=1, max_wait_seconds=1)
        assert result is True

        # Check that the token was consumed
        stats = limiter.get_stats()
        assert stats["available_request_tokens"] == 9

    def test_cost_calculation(self) -> None:
        """Cost is calculated correctly."""
        config = RateLimitConfig(
            cost_per_input_token=0.01,
            cost_per_output_token=0.02,
        )
        limiter = RateLimiter(config)

        # 100 input tokens = $0.001, 100 output tokens = $0.002
        limiter.acquire(input_tokens=100, output_tokens=100)
        limiter.record_usage(input_tokens=100, output_tokens=100)

        stats = limiter.get_stats()
        assert stats["total_cost"] == pytest.approx(0.003, abs=0.0001)

    def test_multiple_requests(self) -> None:
        """Multiple requests work correctly."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        # Make 10 requests (uses all tokens)
        for _ in range(10):
            assert limiter.acquire(num_requests=1)

        # 11th should fail
        assert not limiter.acquire(num_requests=1)

    def test_acquire_multiple_requests(self) -> None:
        """Acquire with multiple requests works."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        # Acquire 5 requests at once
        assert limiter.acquire(num_requests=5)

        # Should have 5 left
        for _ in range(5):
            assert limiter.acquire(num_requests=1)

        # Should be exhausted
        assert not limiter.acquire(num_requests=1)

    def test_thread_safety(self) -> None:
        """RateLimiter is thread-safe."""
        import threading

        config = RateLimitConfig(requests_per_minute=100)
        limiter = RateLimiter(config)

        successes = []
        failures = []

        def worker() -> None:
            for _ in range(20):
                if limiter.acquire(num_requests=1):
                    successes.append(1)
                else:
                    failures.append(1)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 100 requests available, 100 total attempted
        assert len(successes) == 100
        assert len(failures) == 0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_basic_recording(self) -> None:
        """Basic cost recording."""
        tracker = CostTracker()

        tracker.record_request(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            cost_per_input_1k=0.03,
            cost_per_output_1k=0.06,
        )

        # 100/1000 * 0.03 = 0.003
        # 200/1000 * 0.06 = 0.012
        # Total = 0.015
        cost = tracker.get_total_cost()
        assert cost == pytest.approx(0.015, abs=0.0001)

    def test_cost_by_provider(self) -> None:
        """Cost tracking by provider."""
        tracker = CostTracker()

        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)
        tracker.record_request("gemini", "1.5-pro", 100, 100, 0.001, 0.002)

        costs = tracker.get_cost_by_provider()
        assert "openai" in costs
        assert "gemini" in costs
        assert costs["openai"] > costs["gemini"]

    def test_cost_by_model(self) -> None:
        """Cost tracking by model."""
        tracker = CostTracker()

        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)
        tracker.record_request("openai", "gpt-3.5", 100, 100, 0.001, 0.002)

        costs = tracker.get_cost_by_model()
        assert "openai/gpt-4" in costs
        assert "openai/gpt-3.5" in costs
        assert costs["openai/gpt-4"] > costs["openai/gpt-3.5"]

    def test_daily_cost(self) -> None:
        """Daily cost tracking."""
        tracker = CostTracker()

        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)

        today = datetime.now().strftime("%Y-%m-%d")
        cost = tracker.get_daily_cost(today)
        assert cost > 0

        # Unknown date should return 0
        cost_unknown = tracker.get_daily_cost("2099-12-31")
        assert cost_unknown == 0.0

    def test_cost_summary(self) -> None:
        """Cost summary includes all info."""
        tracker = CostTracker()

        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)
        tracker.record_request("gemini", "1.5-pro", 100, 100, 0.001, 0.002)

        summary = tracker.get_cost_summary()
        assert "total_cost" in summary
        assert "cost_today" in summary
        assert "cost_this_month" in summary
        assert "by_provider" in summary
        assert summary["total_cost"] > 0

    def test_multiple_records_same_provider(self) -> None:
        """Multiple records for same provider accumulate."""
        tracker = CostTracker()

        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)
        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)

        cost = tracker.get_cost_by_provider()["openai"]
        # Should be double the single record cost
        expected = pytest.approx(0.003 * 2, abs=0.0001)
        assert cost == expected

    def test_thread_safety(self) -> None:
        """CostTracker is thread-safe."""
        import threading

        tracker = CostTracker()

        def worker() -> None:
            for _ in range(10):
                tracker.record_request(
                    "openai",
                    "gpt-4",
                    100,
                    100,
                    0.01,
                    0.02,
                )

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 50 total records (5 threads * 10 records each)
        summary = tracker.get_cost_summary()
        assert summary["total_cost"] > 0

    def test_zero_cost(self) -> None:
        """Zero cost for free providers."""
        tracker = CostTracker()

        tracker.record_request(
            "ollama",
            "llama3",
            1000,
            1000,
            cost_per_input_1k=0.0,
            cost_per_output_1k=0.0,
        )

        cost = tracker.get_total_cost()
        assert cost == 0.0

    def test_monthly_cost(self) -> None:
        """Monthly cost calculation."""
        tracker = CostTracker()

        # Record for today
        tracker.record_request("openai", "gpt-4", 100, 100, 0.01, 0.02)

        summary = tracker.get_cost_summary()
        assert summary["cost_today"] > 0
        assert summary["cost_this_month"] > 0
        assert summary["cost_this_month"] >= summary["cost_today"]
