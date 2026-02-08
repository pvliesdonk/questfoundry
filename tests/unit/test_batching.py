"""Tests for pipeline.batching module."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from questfoundry.pipeline.batching import batch_llm_calls, is_connectivity_error

# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_empty_list() -> None:
    """Empty input returns empty results."""

    async def _noop(item: str) -> tuple[str, int, int]:
        return item, 1, 10  # pragma: no cover

    results, calls, tokens, errors = await batch_llm_calls([], _noop)
    assert results == []
    assert calls == 0
    assert tokens == 0
    assert errors == []


@pytest.mark.asyncio
async def test_batch_single_item() -> None:
    """Single item works correctly."""

    async def _echo(item: str) -> tuple[str, int, int]:
        return item.upper(), 1, 100

    results, calls, tokens, errors = await batch_llm_calls(["hello"], _echo)
    assert results == ["HELLO"]
    assert calls == 1
    assert tokens == 100
    assert errors == []


@pytest.mark.asyncio
async def test_batch_preserves_order() -> None:
    """Results are in input order regardless of completion order."""
    completion_order: list[int] = []

    async def _delayed(item: tuple[int, float]) -> tuple[int, int, int]:
        idx, delay = item
        await asyncio.sleep(delay)
        completion_order.append(idx)
        return idx * 10, 1, idx

    # Item 0 takes longest, item 2 shortest
    items = [(0, 0.03), (1, 0.02), (2, 0.01)]
    results, calls, tokens, errors = await batch_llm_calls(items, _delayed, max_concurrency=3)

    # Results in input order
    assert results == [0, 10, 20]
    assert calls == 3
    assert tokens == 3  # 0 + 1 + 2
    assert errors == []
    # Completion order should be reversed (shortest first)
    assert completion_order == [2, 1, 0]


@pytest.mark.asyncio
async def test_batch_respects_concurrency() -> None:
    """Semaphore limits concurrent calls."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def _track_concurrency(item: int) -> tuple[int, int, int]:
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent
        await asyncio.sleep(0.02)
        async with lock:
            current_concurrent -= 1
        return item, 1, 0

    items = list(range(6))
    results, calls, _tokens, errors = await batch_llm_calls(
        items, _track_concurrency, max_concurrency=2
    )

    assert max_concurrent <= 2
    assert len([r for r in results if r is not None]) == 6
    assert calls == 6
    assert errors == []


@pytest.mark.asyncio
async def test_batch_collects_errors() -> None:
    """Failed items produce None results and appear in errors list."""

    async def _fail_on_odd(item: int) -> tuple[int, int, int]:
        if item % 2 == 1:
            msg = f"odd number: {item}"
            raise ValueError(msg)
        return item * 10, 1, 10

    items = [0, 1, 2, 3, 4]
    results, calls, tokens, errors = await batch_llm_calls(items, _fail_on_odd)

    assert results[0] == 0
    assert results[1] is None
    assert results[2] == 20
    assert results[3] is None
    assert results[4] == 40
    assert calls == 3  # Only successful calls counted
    assert tokens == 30
    assert len(errors) == 2
    error_indices = {idx for idx, _ in errors}
    assert error_indices == {1, 3}


@pytest.mark.asyncio
async def test_batch_fail_fast() -> None:
    """fail_fast=True cancels remaining tasks on first error."""
    calls_made: list[int] = []

    async def _fail_second(item: int) -> tuple[int, int, int]:
        calls_made.append(item)
        if item == 1:
            msg = "fail"
            raise ValueError(msg)
        await asyncio.sleep(0.05)  # Give time for item 1 to fail
        return item, 1, 0

    # With max_concurrency=1, items run sequentially
    items = [0, 1, 2, 3]
    results, _calls, _tokens, errors = await batch_llm_calls(
        items, _fail_second, max_concurrency=1, fail_fast=True
    )

    assert len(errors) >= 1
    # Item 2 and 3 should not have completed successfully
    assert results[2] is None or results[3] is None


@pytest.mark.asyncio
async def test_batch_aggregates_calls_and_tokens() -> None:
    """LLM calls and tokens are summed across all items."""

    async def _varied(item: int) -> tuple[str, int, int]:
        return f"r{item}", item + 1, (item + 1) * 100

    items = [0, 1, 2]
    results, calls, tokens, errors = await batch_llm_calls(items, _varied)

    assert results == ["r0", "r1", "r2"]
    assert calls == 1 + 2 + 3  # 6
    assert tokens == 100 + 200 + 300  # 600
    assert errors == []


# ---------------------------------------------------------------------------
# is_connectivity_error tests
# ---------------------------------------------------------------------------


class TestIsConnectivityError:
    """Tests for the is_connectivity_error classifier."""

    def test_httpx_connect_error(self) -> None:
        """httpx.ConnectError is a connectivity error."""
        exc = httpx.ConnectError("All connection attempts failed")
        assert is_connectivity_error(exc) is True

    def test_httpx_read_timeout(self) -> None:
        """httpx.ReadTimeout is a connectivity error."""
        exc = httpx.ReadTimeout("timed out")
        assert is_connectivity_error(exc) is True

    def test_httpx_connect_timeout(self) -> None:
        """httpx.ConnectTimeout is a connectivity error."""
        exc = httpx.ConnectTimeout("timed out")
        assert is_connectivity_error(exc) is True

    def test_python_connection_error(self) -> None:
        """Python built-in ConnectionError is a connectivity error."""
        exc = ConnectionRefusedError("Connection refused")
        assert is_connectivity_error(exc) is True

    def test_validation_error_is_not_connectivity(self) -> None:
        """ValueError is NOT a connectivity error."""
        assert is_connectivity_error(ValueError("bad data")) is False

    def test_runtime_error_is_not_connectivity(self) -> None:
        """RuntimeError is NOT a connectivity error."""
        assert is_connectivity_error(RuntimeError("something broke")) is False

    def test_wrapped_cause_single_level(self) -> None:
        """LangChain-wrapped httpx error detected via __cause__."""
        inner = httpx.ConnectError("All connection attempts failed")
        outer = RuntimeError("invocation failed")
        outer.__cause__ = inner
        assert is_connectivity_error(outer) is True

    def test_wrapped_cause_double_level(self) -> None:
        """Double-wrapped httpx error detected via nested __cause__."""
        innermost = httpx.ConnectError("refused")
        middle = RuntimeError("retry failed")
        middle.__cause__ = innermost
        outer = RuntimeError("LLM call failed")
        outer.__cause__ = middle
        assert is_connectivity_error(outer) is True

    def test_wrapped_non_connectivity_cause(self) -> None:
        """Wrapped ValueError cause is NOT connectivity."""
        inner = ValueError("bad schema")
        outer = RuntimeError("invocation failed")
        outer.__cause__ = inner
        assert is_connectivity_error(outer) is False


# ---------------------------------------------------------------------------
# Connectivity retry tests
# ---------------------------------------------------------------------------


class TestBatchConnectivityRetry:
    """Tests for the connectivity retry loop in batch_llm_calls."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """All items fail with ConnectError, hook says retry, second attempt works."""
        attempt = 0

        async def _fail_then_succeed(item: int) -> tuple[int, int, int]:
            nonlocal attempt
            if attempt == 0:
                raise httpx.ConnectError("server down")
            return item * 10, 1, 100

        hook_calls: list[tuple[int, int]] = []

        async def _hook(failed: int, total: int, _msg: str) -> bool:
            nonlocal attempt
            hook_calls.append((failed, total))
            attempt = 1  # Mark server as "back up"
            return True

        items = [1, 2, 3]
        results, calls, _tokens, errors = await batch_llm_calls(
            items, _fail_then_succeed, on_connectivity_error=_hook
        )

        assert results == [10, 20, 30]
        assert errors == []
        assert calls == 3
        assert len(hook_calls) == 1
        assert hook_calls[0] == (3, 3)

    @pytest.mark.asyncio
    async def test_retry_declined(self) -> None:
        """All items fail, hook returns False — errors are preserved."""

        async def _always_fail(_item: int) -> tuple[int, int, int]:
            raise httpx.ConnectError("server down")

        async def _decline(_f: int, _t: int, _m: str) -> bool:
            return False

        items = [1, 2, 3]
        results, _calls, _tokens, errors = await batch_llm_calls(
            items, _always_fail, on_connectivity_error=_decline
        )

        assert all(r is None for r in results)
        assert len(errors) == 3

    @pytest.mark.asyncio
    async def test_no_hook_no_retry(self) -> None:
        """Without on_connectivity_error, batch failures are not retried."""

        async def _always_fail(_item: int) -> tuple[int, int, int]:
            raise httpx.ConnectError("server down")

        items = [1, 2, 3]
        results, _calls, _tokens, errors = await batch_llm_calls(items, _always_fail)

        assert all(r is None for r in results)
        assert len(errors) == 3

    @pytest.mark.asyncio
    async def test_max_retries_respected(self) -> None:
        """Retry loop stops after _MAX_CONNECTIVITY_RETRIES attempts."""
        hook_calls = 0

        async def _always_fail(_item: int) -> tuple[int, int, int]:
            raise httpx.ConnectError("server down")

        async def _always_retry(_f: int, _t: int, _m: str) -> bool:
            nonlocal hook_calls
            hook_calls += 1
            return True

        items = [1, 2, 3]
        results, _calls, _tokens, errors = await batch_llm_calls(
            items, _always_fail, on_connectivity_error=_always_retry
        )

        # 3 retries max (from _MAX_CONNECTIVITY_RETRIES)
        assert hook_calls == 3
        assert all(r is None for r in results)
        assert len(errors) == 3

    @pytest.mark.asyncio
    async def test_mixed_errors_no_retry(self) -> None:
        """Mix of connectivity and validation errors — no retry triggered."""
        hook_called = False

        async def _mixed_fail(item: int) -> tuple[int, int, int]:
            if item % 2 == 0:
                raise httpx.ConnectError("server down")
            raise ValueError("bad data")

        async def _hook(_f: int, _t: int, _m: str) -> bool:
            nonlocal hook_called
            hook_called = True  # pragma: no cover
            return True  # pragma: no cover

        items = [0, 1, 2, 3]
        results, _calls, _tokens, errors = await batch_llm_calls(
            items, _mixed_fail, on_connectivity_error=_hook
        )

        assert not hook_called
        assert all(r is None for r in results)
        assert len(errors) == 4

    @pytest.mark.asyncio
    async def test_single_connectivity_error_no_retry(self) -> None:
        """Only 1 item fails with ConnectError (< threshold of 2) — no retry."""
        hook_called = False

        async def _one_fails(item: int) -> tuple[int, int, int]:
            if item == 0:
                raise httpx.ConnectError("server blip")
            return item * 10, 1, 10

        async def _hook(_f: int, _t: int, _m: str) -> bool:
            nonlocal hook_called
            hook_called = True  # pragma: no cover
            return True  # pragma: no cover

        items = [0, 1, 2]
        results, _calls, _tokens, errors = await batch_llm_calls(
            items, _one_fails, on_connectivity_error=_hook
        )

        assert not hook_called
        assert results[0] is None
        assert results[1] == 10
        assert results[2] == 20
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_retry_only_failed_items(self) -> None:
        """On retry, only previously failed items are retried — successful ones kept."""
        attempt_counts: dict[int, int] = {}

        async def _fail_first_two(item: int) -> tuple[int, int, int]:
            attempt_counts[item] = attempt_counts.get(item, 0) + 1
            # Items 0 and 1 fail on first attempt, succeed on retry
            if item < 2 and attempt_counts[item] == 1:
                raise httpx.ConnectError("server down")
            return item * 10, 1, 100

        async def _retry(_f: int, _t: int, _m: str) -> bool:
            return True

        items = [0, 1, 2]
        results, _calls, _tokens, errors = await batch_llm_calls(
            items, _fail_first_two, on_connectivity_error=_retry
        )

        assert results == [0, 10, 20]
        assert errors == []
        # Item 2 should only have been called once (it succeeded first time)
        assert attempt_counts[2] == 1
        # Items 0 and 1 should have been called twice
        assert attempt_counts[0] == 2
        assert attempt_counts[1] == 2
