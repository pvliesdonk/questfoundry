"""Rate limiting and cost tracking for QuestFoundry providers."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Implements token bucket algorithm with support for multiple
    limit types: requests per minute, tokens per hour, and optional
    cost limits per day.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        tokens_per_hour: Maximum tokens allowed per hour
        cost_per_day: Optional maximum cost per day (in USD)
        cost_per_input_token: Cost per 1000 input tokens
        cost_per_output_token: Cost per 1000 output tokens
    """

    requests_per_minute: int = 60
    tokens_per_hour: int = 1000
    cost_per_day: Optional[float] = None
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0


class RateLimiter:
    """Token bucket rate limiter with cost tracking.

    Implements the token bucket algorithm to enforce rate limits.
    Supports three types of limits:
    1. Request rate (per minute)
    2. Token rate (per hour)
    3. Cost limits (per day) - optional

    The limiter uses refillable tokens: tokens are added over time
    at a configured rate, and requests consume tokens. This allows
    handling bursts while respecting long-term rate limits.

    Example:
        ```python
        config = RateLimitConfig(
            requests_per_minute=90,
            tokens_per_hour=90000,
        )
        limiter = RateLimiter(config)

        # Check if we can make a request
        if limiter.acquire(input_tokens=100, output_tokens=250):
            response = provider.generate_text(prompt)
            limiter.record_usage(100, 250)
        else:
            raise RateLimitError("Rate limit exceeded")
        ```
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limiting configuration
        """
        logger.debug(
            (
                "Initializing RateLimiter with config: "
                "requests_per_minute=%d, tokens_per_hour=%d"
            ),
            config.requests_per_minute,
            config.tokens_per_hour,
        )
        self.config = config

        # Token buckets (start full)
        self.request_tokens = float(config.requests_per_minute)
        self.token_tokens = float(config.tokens_per_hour)
        self.cost_tokens = (
            config.cost_per_day * 100.0 if config.cost_per_day else None
        )  # Convert to cents

        # Last refill times
        self.last_request_refill = time.time()
        self.last_token_refill = time.time()
        self.last_cost_refill = time.time()

        # Thread safety
        self._lock = Lock()

        # Usage tracking
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        if config.cost_per_day:
            logger.trace("Cost limit configured: $%s per day", config.cost_per_day)
        else:
            logger.trace("No cost limit configured")
        logger.trace("RateLimiter initialized successfully")

    def _refill_buckets(self) -> None:
        """Refill token buckets based on elapsed time.

        This is called before every check to update token counts.
        """
        now = time.time()

        # Refill request tokens (per minute)
        minutes_elapsed = (now - self.last_request_refill) / 60.0
        self.request_tokens = min(
            float(self.config.requests_per_minute),
            self.request_tokens + self.config.requests_per_minute * minutes_elapsed,
        )
        self.last_request_refill = now

        # Refill token bucket (per hour)
        hours_elapsed = (now - self.last_token_refill) / 3600.0
        self.token_tokens = min(
            float(self.config.tokens_per_hour),
            self.token_tokens + self.config.tokens_per_hour * hours_elapsed,
        )
        self.last_token_refill = now

        # Refill cost bucket (per day)
        if self.cost_tokens is not None:
            days_elapsed = (now - self.last_cost_refill) / 86400.0
            if self.config.cost_per_day:
                max_cost_tokens = self.config.cost_per_day * 100.0
            else:
                max_cost_tokens = 0
            self.cost_tokens = min(
                max_cost_tokens,
                self.cost_tokens + max_cost_tokens * days_elapsed,
            )
            self.last_cost_refill = now

    def acquire(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_requests: int = 1,
    ) -> bool:
        """Acquire tokens for a request.

        Returns True if all limits allow the request and consumes tokens.
        Returns False if any limit would be exceeded without consuming tokens.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_requests: Number of requests (default: 1)

        Returns:
            True if request allowed, False if rate limited
        """
        logger.trace(
            "Attempting to acquire rate limit tokens: requests=%d, tokens=%d",
            num_requests,
            input_tokens + output_tokens,
        )

        with self._lock:
            self._refill_buckets()

            total_tokens = input_tokens + output_tokens

            # Check request limit
            if self.request_tokens < num_requests:
                logger.warning(
                    "Request rate limit exceeded: available=%.1f, required=%d",
                    self.request_tokens,
                    num_requests,
                )
                return False

            # Check token limit
            if self.token_tokens < total_tokens:
                logger.warning(
                    "Token rate limit exceeded: available=%.1f, required=%d",
                    self.token_tokens,
                    total_tokens,
                )
                return False

            # Check cost limit (approximate, in cents)
            estimated_cost_cents = (
                self._estimate_cost(input_tokens, output_tokens) * 100
            )
            if self.cost_tokens is not None and self.cost_tokens < estimated_cost_cents:
                logger.warning(
                    "Cost limit exceeded: available=%.2f, required=%.2f cents",
                    self.cost_tokens,
                    estimated_cost_cents,
                )
                return False

            # All checks passed, consume tokens
            logger.trace("Rate limit check passed, consuming tokens")
            self.request_tokens -= num_requests
            self.token_tokens -= total_tokens
            if self.cost_tokens is not None:
                self.cost_tokens -= estimated_cost_cents
            logger.trace("Tokens consumed successfully")

            return True

    def check_limit(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_requests: int = 1,
    ) -> bool:
        """Check if request would be allowed (non-consuming check).

        Similar to acquire() but doesn't consume tokens.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_requests: Number of requests (default: 1)

        Returns:
            True if request would be allowed, False if rate limited
        """
        with self._lock:
            self._refill_buckets()

            total_tokens = input_tokens + output_tokens

            # Check all limits
            if self.request_tokens < num_requests:
                return False

            if self.token_tokens < total_tokens:
                return False

            estimated_cost_cents = (
                self._estimate_cost(input_tokens, output_tokens) * 100
            )
            if self.cost_tokens is not None and self.cost_tokens < estimated_cost_cents:
                return False

            return True

    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_requests: int = 1,
    ) -> None:
        """Record actual usage for tracking.

        This is separate from acquire() to allow tracking actual
        usage separately from token consumption.

        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens used
            num_requests: Number of requests made
        """
        with self._lock:
            self.total_requests += num_requests
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            cost = self._calculate_cost(input_tokens, output_tokens)
            self.total_cost += cost

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost before making request (USD)."""
        return self._calculate_cost(input_tokens, output_tokens)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost of tokens (USD)."""
        input_cost = (input_tokens / 1000.0) * self.config.cost_per_input_token
        output_cost = (output_tokens / 1000.0) * self.config.cost_per_output_token
        return input_cost + output_cost

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with usage stats and available tokens
        """
        with self._lock:
            self._refill_buckets()

            return {
                "total_requests": self.total_requests,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost": round(self.total_cost, 4),
                "available_request_tokens": int(self.request_tokens),
                "available_token_tokens": int(self.token_tokens),
            }

    def wait_until_available(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_requests: int = 1,
        max_wait_seconds: int = 3600,
    ) -> bool:
        """Wait until rate limit allows the request.

        Useful for batch operations. Blocks until tokens available
        or max_wait_seconds elapsed.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_requests: Number of requests
            max_wait_seconds: Maximum wait time (default: 1 hour)

        Returns:
            True if tokens acquired, False if max wait exceeded
        """
        logger.debug(
            "Waiting for rate limit tokens available (max_wait=%d seconds)",
            max_wait_seconds,
        )
        start_time = time.time()
        wait_count = 0

        while True:
            if self.acquire(input_tokens, output_tokens, num_requests):
                elapsed = time.time() - start_time
                logger.info(
                    "Rate limit tokens acquired after %.2f seconds (waited %d times)",
                    elapsed,
                    wait_count,
                )
                return True

            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                logger.warning("Max wait time exceeded for rate limit tokens")
                return False

            # Exponential backoff: 1s, 2s, 4s, max 10s
            wait_time = min(2 ** int(elapsed // 60), 10)
            wait_count += 1
            logger.trace(
                "Rate limited, waiting %.1f seconds before retry (attempt %d)",
                wait_time,
                wait_count,
            )
            time.sleep(wait_time)


class CostTracker:
    """Track API costs across providers and time periods.

    Maintains detailed cost records broken down by:
    - Provider (OpenAI, Gemini, etc.)
    - Model (gpt-4, gpt-3.5-turbo, etc.)
    - Time period (daily, monthly, total)

    Example:
        ```python
        tracker = CostTracker()
        tracker.record_request(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=250,
            cost_per_input_1k=0.03,
            cost_per_output_1k=0.06,
        )
        print(f"Total cost: ${tracker.get_total_cost():.2f}")
        print(f"Cost by provider: {tracker.get_cost_by_provider()}")
        ```
    """

    def __init__(self, history_days: int = 30) -> None:
        """Initialize cost tracker.

        Args:
            history_days: How many days of history to keep
        """
        logger.debug("Initializing CostTracker with history_days=%d", history_days)
        self.history_days = history_days
        self.costs_by_provider: dict[str, float] = defaultdict(float)
        self.costs_by_date: dict[str, float] = defaultdict(float)
        self.costs_by_model: dict[str, float] = defaultdict(float)
        self._lock = Lock()
        logger.trace("CostTracker initialized successfully")

    def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_input_1k: float,
        cost_per_output_1k: float,
    ) -> None:
        """Record a request and its cost.

        Args:
            provider: Provider name (e.g., 'openai')
            model: Model name (e.g., 'gpt-4')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_per_input_1k: Cost per 1000 input tokens (USD)
            cost_per_output_1k: Cost per 1000 output tokens (USD)
        """
        cost = (input_tokens / 1000.0) * cost_per_input_1k + (
            output_tokens / 1000.0
        ) * cost_per_output_1k

        date_key = datetime.now().strftime("%Y-%m-%d")

        logger.trace(
            "Recording cost for %s/%s: tokens=%d+%d, cost=$%.4f",
            provider,
            model,
            input_tokens,
            output_tokens,
            cost,
        )

        with self._lock:
            self.costs_by_provider[provider] += cost
            self.costs_by_date[date_key] += cost
            self.costs_by_model[f"{provider}/{model}"] += cost

        logger.debug(
            "Cost recorded for %s/%s: $%.4f (total so far: $%.4f)",
            provider,
            model,
            cost,
            self.costs_by_provider[provider],
        )

    def get_total_cost(self) -> float:
        """Get total cost across all time.

        Returns:
            Total cost in USD
        """
        with self._lock:
            return sum(self.costs_by_provider.values())

    def get_cost_by_provider(self) -> dict[str, float]:
        """Get costs broken down by provider.

        Returns:
            Dictionary mapping provider names to costs
        """
        with self._lock:
            return dict(self.costs_by_provider)

    def get_cost_by_model(self) -> dict[str, float]:
        """Get costs broken down by provider and model.

        Returns:
            Dictionary mapping 'provider/model' to costs
        """
        with self._lock:
            return dict(self.costs_by_model)

    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get cost for a specific day.

        Args:
            date: Date in YYYY-MM-DD format (defaults to today)

        Returns:
            Cost for that day in USD
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            return self.costs_by_date.get(date, 0.0)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get comprehensive cost summary.

        Returns:
            Dictionary with total, daily, monthly costs and breakdown
        """
        logger.trace("Generating cost summary")

        with self._lock:
            total = sum(self.costs_by_provider.values())
            today = datetime.now().strftime("%Y-%m-%d")
            current_month = datetime.now().strftime("%Y-%m")

            month_cost = sum(
                v for k, v in self.costs_by_date.items() if k.startswith(current_month)
            )

            providers_summary = {
                k: round(v, 4) for k, v in self.costs_by_provider.items()
            }

            summary = {
                "total_cost": round(total, 4),
                "cost_today": round(self.costs_by_date.get(today, 0.0), 4),
                "cost_this_month": round(month_cost, 4),
                "by_provider": providers_summary,
            }

        logger.debug(
            "Cost summary: total=$%.4f, today=$%.4f, this_month=$%.4f",
            summary["total_cost"],
            summary["cost_today"],
            summary["cost_this_month"],
        )
        logger.trace("Cost breakdown by provider: %s", providers_summary)

        return summary


class RateLimitError(Exception):
    """Raised when rate limit would be exceeded."""

    pass


class CostLimitExceededError(Exception):
    """Raised when cost limit exceeded."""

    pass
