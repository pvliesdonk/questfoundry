"""Rate-limit classification and backoff helpers for LLM providers.

The classifier and helper here exist so transport-level HTTP 429 errors do
not burn a slot in the per-stage *semantic* retry budget. Issue #1581: a bare
``except Exception`` in the validation/repair loops was routing rate-limit
errors through the schema-repair path, appending an apology ``HumanMessage``
and immediately retrying with no backoff — making the rate-limit pressure
worse and exhausting the retry budget on transport errors.

The module exports two pieces:

- ``classify_rate_limit_error(exc)`` — walk the exception ``__cause__`` /
  ``__context__`` chain and return a ``ProviderRateLimitError`` (with
  ``retry_after_seconds``) if any layer looks like a rate-limit. Returns
  ``None`` otherwise.
- ``ainvoke_with_rate_limit_retry(model, messages, *, config)`` — wrap a
  LangChain ``model.ainvoke`` call. On rate-limit, sleep and retry without
  consuming the caller's semantic-retry budget. On any other exception,
  re-raise immediately. After exceeding the total wait budget, raise
  ``ProviderRateLimitError`` so the caller surfaces a hard transport failure.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

from questfoundry.observability.logging import get_logger
from questfoundry.providers.base import ProviderRateLimitError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.messages import BaseMessage

log = get_logger(__name__)


SINGLE_WAIT_CAP_SECONDS: float = 60.0
"""Maximum seconds to sleep on any single backoff. Caps a runaway header
value (provider could legally say "wait 600s") so the helper never blocks
a stage for that long without the caller getting a chance to surface it."""

TOTAL_WAIT_CAP_SECONDS: float = 300.0
"""Maximum cumulative sleep across one ``ainvoke`` call. Beyond this we give
up and raise ``ProviderRateLimitError`` so the caller can decide what to do
(typically: fail the stage cleanly, with a message that says transport, not
schema)."""

DEFAULT_INITIAL_WAIT_SECONDS: float = 5.0
"""Base delay for exponential backoff when the provider sends a 429 with no
``Retry-After`` header. Doubles on each consecutive rate-limit until either
``SINGLE_WAIT_CAP_SECONDS`` is hit or we exit."""

_MAX_BACKOFF_ATTEMPTS: int = 12
"""Hard ceiling on backoff loop iterations — prevents an infinite loop when
both the total-wait cap math and the provider behave pathologically."""

_JITTER_FRACTION: float = 0.10
"""±10% jitter applied to every sleep so parallel callers don't stampede the
same reset window."""


def classify_rate_limit_error(exc: BaseException) -> ProviderRateLimitError | None:
    """Return a ``ProviderRateLimitError`` if ``exc`` (or any wrapped cause)
    is a rate-limit, else ``None``.

    Recognises:
    - ``anthropic.RateLimitError`` and ``openai.RateLimitError`` (both expose
      a ``.response`` attribute with status 429 and a ``Retry-After`` header).
    - ``google.api_core.exceptions.ResourceExhausted`` by class name.
    - Any exception with a ``.response`` whose ``.status_code == 429``
      (catches ``httpx.HTTPStatusError`` and any LangChain wrapper that
      preserves ``.response``).
    - Any exception with ``.status_code == 429`` directly on the instance.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        info = _classify_single(cur)
        if info is not None:
            return info
        cur = cur.__cause__ or cur.__context__
    return None


def _classify_single(exc: BaseException) -> ProviderRateLimitError | None:
    """Classify one exception (no chain walk)."""
    name = type(exc).__name__
    module = type(exc).__module__

    # google.api_core.exceptions.ResourceExhausted — class-name match because
    # the package is optional in this project (only present when google extras
    # are installed). Gate on module prefix so a third-party class that happens
    # to share the name isn't misclassified as a Google rate-limit.
    if name == "ResourceExhausted" and (
        module.startswith("google") or module.startswith("langchain_google")
    ):
        return ProviderRateLimitError("google", str(exc) or "ResourceExhausted")

    # Provider-specific RateLimitError classes (anthropic + openai both name
    # them this and both expose .response). Match by name + module-prefix to
    # avoid colliding with any future custom subclass that happens to share
    # the suffix.
    if name == "RateLimitError" and (module.startswith("anthropic") or module.startswith("openai")):
        provider = _provider_from_module(module)
        retry_after = _read_retry_after(getattr(exc, "response", None))
        return ProviderRateLimitError(provider, str(exc) or "rate_limit", retry_after)

    # Generic 429 via response object (httpx.HTTPStatusError, LangChain wraps).
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        provider = _provider_from_module(module)
        retry_after = _read_retry_after(response)
        return ProviderRateLimitError(provider, str(exc) or "rate_limit", retry_after)

    # status_code on the exception itself (some thin wrappers).
    if getattr(exc, "status_code", None) == 429:
        provider = _provider_from_module(module)
        return ProviderRateLimitError(provider, str(exc) or "rate_limit")

    return None


def _provider_from_module(module: str) -> str:
    """Map a Python module name to a provider tag (best-effort)."""
    if module.startswith("anthropic") or module.startswith("langchain_anthropic"):
        return "anthropic"
    if module.startswith("openai") or module.startswith("langchain_openai"):
        return "openai"
    if module.startswith("google") or module.startswith("langchain_google"):
        return "google"
    return "unknown"


def _read_retry_after(response: Any) -> float | None:
    """Extract ``Retry-After`` seconds from a response's headers, if present.

    Provider quirks the helper ignores deliberately:
    - HTTP-date format for ``Retry-After`` (e.g. ``Wed, 21 Oct 2026 07:28:00 GMT``)
      — accepted by the spec but rare in practice for 429s. We fall back to
      our exponential default if the value isn't a parseable float.
    - Anthropic's ``anthropic-ratelimit-*-reset`` headers — expressed as
      seconds-until-reset *or* an absolute timestamp depending on header.
      ``Retry-After`` is always present alongside on a 429, so reading just
      that is sufficient and avoids per-provider parsing.
    """
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _next_backoff(retry_after_hint: float | None, consecutive: int) -> float:
    """Compute the next sleep duration, with jitter, respecting the cap."""
    if retry_after_hint is not None and retry_after_hint > 0:
        base = retry_after_hint
    else:
        # 5, 10, 20, 40, 60, 60... seconds
        base = DEFAULT_INITIAL_WAIT_SECONDS * (2 ** max(0, consecutive - 1))

    capped = min(base, SINGLE_WAIT_CAP_SECONDS)
    jitter = capped * _JITTER_FRACTION * (random.random() * 2 - 1)
    return max(0.0, capped + jitter)


async def ainvoke_with_rate_limit_retry(
    model: Any,
    messages: Sequence[BaseMessage],
    *,
    config: Any = None,
) -> Any:
    """Invoke ``model.ainvoke(messages, config=config)`` with rate-limit-aware
    retry.

    Behaviour:
    - On success, return the result.
    - On a classified rate-limit error, sleep ``retry_after`` seconds (jittered,
      capped at ``SINGLE_WAIT_CAP_SECONDS``) and retry. This does NOT count
      toward the caller's semantic-retry budget — that budget is reserved for
      schema/validation failures.
    - On any other exception, re-raise immediately so the caller's existing
      ``except ValidationError`` / ``except Exception`` paths run unchanged.
    - If cumulative sleep would exceed ``TOTAL_WAIT_CAP_SECONDS``, raise the
      classified ``ProviderRateLimitError`` so the caller surfaces a clear
      transport-level failure (and does not append a schema-repair message).
    """
    total_waited = 0.0
    consecutive = 0
    last_classified: ProviderRateLimitError | None = None

    for attempt in range(1, _MAX_BACKOFF_ATTEMPTS + 1):
        try:
            return await model.ainvoke(messages, config=config)
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            classified = classify_rate_limit_error(exc)
            if classified is None:
                raise
            consecutive += 1
            last_classified = classified
            wait = _next_backoff(classified.retry_after_seconds, consecutive)

            if total_waited + wait > TOTAL_WAIT_CAP_SECONDS:
                log.error(
                    "rate_limit_total_wait_exceeded",
                    provider=classified.provider,
                    total_waited=total_waited,
                    next_wait=wait,
                    cap=TOTAL_WAIT_CAP_SECONDS,
                    attempt=attempt,
                )
                raise classified from exc

            # INFO, not WARNING: the helper detected a 429 and handled it
            # correctly. CLAUDE.md reserves WARNING for "this worked but
            # someone should look at it". The two error-level logs above
            # (``rate_limit_total_wait_exceeded`` / ``_attempts_exhausted``)
            # are the failure paths — those re-raise.
            log.info(
                "rate_limit_backoff",
                provider=classified.provider,
                wait_seconds=round(wait, 2),
                attempt=attempt,
                total_waited=round(total_waited, 2),
                retry_after_hint=classified.retry_after_seconds,
            )
            await asyncio.sleep(wait)
            total_waited += wait

    # Iteration ceiling reached without success.  ``last_classified`` is set
    # because every loop iteration either returns, re-raises a non-rate-limit
    # exception, or assigns ``last_classified`` before retrying — and an empty
    # iteration range is impossible (range(1, 13)).
    log.error(
        "rate_limit_attempts_exhausted",
        attempts=_MAX_BACKOFF_ATTEMPTS,
        total_waited=round(total_waited, 2),
    )
    # Explicit guard rather than ``assert`` — assertions are stripped under
    # ``python -O``, which would turn a broken invariant into ``raise None``
    # (a confusing TypeError instead of a real diagnostic).
    if last_classified is None:  # pragma: no cover — invariant from loop above
        raise RuntimeError(
            f"rate_limit_attempts_exhausted: no classified error captured after "
            f"{_MAX_BACKOFF_ATTEMPTS} iterations"
        )
    raise last_classified
