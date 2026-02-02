"""Tests for pipeline.batching module."""

from __future__ import annotations

import asyncio

import pytest

from questfoundry.pipeline.batching import batch_llm_calls


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
