"""Bounded-concurrency batch helper for LLM calls.

Wraps asyncio.Semaphore to limit concurrent calls per provider.
Preserves input order in results and aggregates call/token counts.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = get_logger(__name__)

T = TypeVar("T")
Item = TypeVar("Item")


async def batch_llm_calls(
    items: list[Item],
    call_fn: Callable[[Item], Awaitable[tuple[T, int, int]]],
    max_concurrency: int = 2,
    *,
    fail_fast: bool = False,
) -> tuple[list[T | None], int, int, list[tuple[int, Exception]]]:
    """Run LLM calls concurrently with bounded parallelism.

    Args:
        items: Input items to process.
        call_fn: Async function taking one item, returning
            (result, llm_calls, tokens_used).
        max_concurrency: Maximum concurrent calls (from ModelInfo).
        fail_fast: If True, cancel remaining tasks on first error.
            If False (default), collect errors and continue.

    Returns:
        Tuple of:
            - results: List in input order (None for failed items).
            - total_llm_calls: Sum of LLM calls across all items.
            - total_tokens: Sum of tokens across all items.
            - errors: List of (index, exception) for failed items.
    """
    if not items:
        return [], 0, 0, []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    results: list[T | None] = [None] * len(items)
    llm_calls_list: list[int] = [0] * len(items)
    tokens_list: list[int] = [0] * len(items)
    errors: list[tuple[int, Exception]] = []
    errors_lock = asyncio.Lock()

    async def _run_one(idx: int, item: Item) -> None:
        async with semaphore:
            try:
                result, calls, tokens = await call_fn(item)
                results[idx] = result
                llm_calls_list[idx] = calls
                tokens_list[idx] = tokens
            except Exception as e:
                async with errors_lock:
                    errors.append((idx, e))
                log.warning(
                    "batch_item_failed",
                    index=idx,
                    error=str(e),
                )
                if fail_fast:
                    raise

    tasks = [asyncio.create_task(_run_one(i, item)) for i, item in enumerate(items)]

    if fail_fast:
        # Cancel remaining on first failure
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    else:
        await asyncio.gather(*tasks, return_exceptions=True)

    total_llm_calls = sum(llm_calls_list)
    total_tokens = sum(tokens_list)

    log.debug(
        "batch_complete",
        total_items=len(items),
        succeeded=len(items) - len(errors),
        failed=len(errors),
        total_llm_calls=total_llm_calls,
        total_tokens=total_tokens,
    )

    return results, total_llm_calls, total_tokens, errors
