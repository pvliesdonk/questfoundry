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

_MAX_CONNECTIVITY_RETRIES = 3


def is_connectivity_error(exc: Exception) -> bool:
    """Check if an exception indicates provider connectivity loss.

    Recognises httpx network/timeout errors, Python built-in
    ConnectionError, and QuestFoundry's ProviderConnectionError.
    Walks the ``__cause__`` chain so LangChain-wrapped errors are
    also detected.
    """
    import httpx

    from questfoundry.providers.base import ProviderConnectionError

    if isinstance(
        exc,
        (
            httpx.NetworkError,  # ConnectError, ReadError, WriteError, CloseError
            httpx.TimeoutException,  # ConnectTimeout, ReadTimeout, PoolTimeout
            ConnectionError,  # Python built-in (Refused, Reset, Aborted)
            ProviderConnectionError,  # QF's own type
        ),
    ):
        return True

    # LangChain may wrap httpx errors — walk the cause chain
    cause = exc.__cause__
    if cause is not None and isinstance(cause, Exception):
        return is_connectivity_error(cause)

    return False


def _is_connectivity_loss(errors: list[tuple[int, Exception]]) -> bool:
    """Return True when all errors are connectivity errors and >= 2 items failed."""
    return len(errors) >= 2 and all(is_connectivity_error(e) for _, e in errors)


async def batch_llm_calls(
    items: list[Item],
    call_fn: Callable[[Item], Awaitable[tuple[T, int, int]]],
    max_concurrency: int = 2,
    *,
    fail_fast: bool = False,
    on_connectivity_error: Callable[[int, int, str], Awaitable[bool]] | None = None,
) -> tuple[list[T | None], int, int, list[tuple[int, Exception]]]:
    """Run LLM calls concurrently with bounded parallelism.

    Args:
        items: Input items to process.
        call_fn: Async function taking one item, returning
            (result, llm_calls, tokens_used).
        max_concurrency: Maximum concurrent calls (from ModelInfo).
        fail_fast: If True, cancel remaining tasks on first error.
            If False (default), collect errors and continue.
        on_connectivity_error: Optional async callback invoked when all
            batch errors are connectivity failures. Receives
            ``(failed_count, total_count, error_sample)`` and returns
            ``True`` to retry failed items or ``False`` to accept the
            failure.  When ``None``, no retry is attempted (backward
            compatible).

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

    async def _run_one(
        idx: int,
        item: Item,
        target_errors: list[tuple[int, Exception]],
        target_lock: asyncio.Lock,
        *,
        raise_on_fail: bool = False,
    ) -> None:
        async with semaphore:
            try:
                result, calls, tokens = await call_fn(item)
                results[idx] = result
                llm_calls_list[idx] = calls
                tokens_list[idx] = tokens
            except Exception as e:
                async with target_lock:
                    target_errors.append((idx, e))
                log.warning(
                    "batch_item_failed",
                    index=idx,
                    error=str(e),
                )
                if raise_on_fail:
                    raise

    tasks = [
        asyncio.create_task(_run_one(i, item, errors, errors_lock, raise_on_fail=fail_fast))
        for i, item in enumerate(items)
    ]

    if fail_fast:
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    else:
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- Connectivity retry loop ---
    retry_count = 0
    while (
        on_connectivity_error is not None
        and retry_count < _MAX_CONNECTIVITY_RETRIES
        and _is_connectivity_loss(errors)
    ):
        sample_error = str(errors[0][1])
        should_retry = await on_connectivity_error(len(errors), len(items), sample_error)
        if not should_retry:
            break

        retry_count += 1
        failed_indices = [idx for idx, _ in errors]
        log.info(
            "batch_connectivity_retry",
            retry=retry_count,
            failed_items=len(failed_indices),
        )

        # Reset errors — retry only the previously failed items
        retry_errors: list[tuple[int, Exception]] = []
        retry_lock = asyncio.Lock()

        retry_tasks = [
            asyncio.create_task(_run_one(idx, items[idx], retry_errors, retry_lock))
            for idx in failed_indices
        ]
        await asyncio.gather(*retry_tasks, return_exceptions=True)
        errors = retry_errors

    # Log connectivity loss at ERROR level for visibility (even without hook)
    if _is_connectivity_loss(errors):
        log.error(
            "batch_connectivity_failure",
            total_items=len(items),
            failed=len(errors),
            error_sample=str(errors[0][1]),
        )

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
