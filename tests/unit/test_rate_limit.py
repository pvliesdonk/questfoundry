"""Tests for providers.rate_limit module — classifier + backoff helper.

Verifies that transport-level rate-limit errors (HTTP 429) are recognized,
backed off respecting Retry-After, and surfaced as ProviderRateLimitError —
NOT routed through the semantic-retry / schema-repair path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from questfoundry.providers.base import ProviderRateLimitError
from questfoundry.providers.rate_limit import (
    DEFAULT_INITIAL_WAIT_SECONDS,
    SINGLE_WAIT_CAP_SECONDS,
    TOTAL_WAIT_CAP_SECONDS,
    ainvoke_with_rate_limit_retry,
    classify_rate_limit_error,
)


def _build_httpx_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    """Build a real httpx.Response so .headers / .status_code behave naturally."""
    return httpx.Response(
        status_code=status,
        headers=headers or {},
        request=httpx.Request("POST", "https://example.test/v1"),
    )


class TestClassifierRecognizesAnthropic:
    def test_anthropic_rate_limit_error(self) -> None:
        import anthropic

        response = _build_httpx_response(429, {"retry-after": "7"})
        exc = anthropic.RateLimitError("rate limited", response=response, body=None)

        classified = classify_rate_limit_error(exc)

        assert classified is not None
        assert classified.provider == "anthropic"
        assert classified.retry_after_seconds == pytest.approx(7.0)

    def test_anthropic_without_retry_after_header(self) -> None:
        import anthropic

        response = _build_httpx_response(429, {})
        exc = anthropic.RateLimitError("rate limited", response=response, body=None)

        classified = classify_rate_limit_error(exc)

        assert classified is not None
        assert classified.provider == "anthropic"
        assert classified.retry_after_seconds is None


class TestClassifierRecognizesOpenAI:
    def test_openai_rate_limit_error(self) -> None:
        import openai

        response = _build_httpx_response(429, {"retry-after": "12"})
        exc = openai.RateLimitError("rate limited", response=response, body=None)

        classified = classify_rate_limit_error(exc)

        assert classified is not None
        assert classified.provider == "openai"
        assert classified.retry_after_seconds == pytest.approx(12.0)


class TestClassifierRecognizesWrappedHttpx:
    def test_wrapped_httpx_status_429(self) -> None:
        """A bare httpx.HTTPStatusError with 429 status is classified as rate-limit."""
        response = _build_httpx_response(429, {"retry-after": "3"})
        exc = httpx.HTTPStatusError("429", request=response.request, response=response)

        classified = classify_rate_limit_error(exc)

        assert classified is not None
        assert classified.retry_after_seconds == pytest.approx(3.0)

    def test_non_429_httpx_status_not_classified(self) -> None:
        """A 500 httpx error is NOT a rate-limit error."""
        response = _build_httpx_response(500)
        exc = httpx.HTTPStatusError("500", request=response.request, response=response)

        assert classify_rate_limit_error(exc) is None


class TestClassifierRecognizesGoogleResourceExhausted:
    def test_google_resource_exhausted_by_class_name_and_module(self) -> None:
        """``google.api_core.exceptions.ResourceExhausted`` is detected by
        class-name + module-prefix gating, without requiring the optional
        ``google-api-core`` package to be installed."""
        fake_cls = type(
            "ResourceExhausted",
            (Exception,),
            {"__module__": "google.api_core.exceptions"},
        )
        exc = fake_cls("quota exceeded")

        classified = classify_rate_limit_error(exc)

        assert classified is not None
        assert classified.provider == "google"

    def test_other_module_with_same_classname_not_classified(self) -> None:
        """A class named ``ResourceExhausted`` from an unrelated module is NOT
        classified — the module-prefix gate is the safeguard."""
        fake_cls = type(
            "ResourceExhausted",
            (Exception,),
            {"__module__": "some.unrelated.package"},
        )
        exc = fake_cls("not actually rate-limited")

        assert classify_rate_limit_error(exc) is None


class TestClassifierWalksExceptionChain:
    def test_walks_cause_chain(self) -> None:
        """Classifier finds rate-limit deeper in __cause__."""
        import anthropic

        response = _build_httpx_response(429, {"retry-after": "4"})
        inner = anthropic.RateLimitError("inner", response=response, body=None)
        try:
            raise RuntimeError("LangChain wrapped this") from inner
        except RuntimeError as outer:
            classified = classify_rate_limit_error(outer)

        assert classified is not None
        assert classified.retry_after_seconds == pytest.approx(4.0)


class TestClassifierIgnoresNonRateLimit:
    def test_validation_error_returns_none(self) -> None:
        from pydantic import BaseModel, ValidationError

        class M(BaseModel):
            x: int

        try:
            M.model_validate({"x": "not-an-int"})
        except ValidationError as exc:
            assert classify_rate_limit_error(exc) is None
        else:  # pragma: no cover
            pytest.fail("Pydantic should have raised ValidationError")

    def test_value_error_returns_none(self) -> None:
        assert classify_rate_limit_error(ValueError("nope")) is None


class TestAinvokeWithRateLimitRetrySucceeds:
    @pytest.mark.asyncio
    async def test_returns_immediately_on_success(self) -> None:
        """When ainvoke succeeds first try, no sleep, return value passes through."""

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                return "ok"

        result = await ainvoke_with_rate_limit_retry(_Model(), [], config=None)
        assert result == "ok"
        assert _Model.calls == 1

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self) -> None:
        """A single rate-limit error sleeps then retries successfully."""
        import anthropic

        response = _build_httpx_response(429, {"retry-after": "0.01"})

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                if _Model.calls == 1:
                    raise anthropic.RateLimitError("rate", response=response, body=None)
                return "ok"

        result = await ainvoke_with_rate_limit_retry(_Model(), [], config=None)
        assert result == "ok"
        assert _Model.calls == 2


class TestAinvokeWithRateLimitRetryReraises:
    @pytest.mark.asyncio
    async def test_non_rate_limit_propagates_immediately(self) -> None:
        """Non-rate-limit exceptions are re-raised without sleep."""

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                raise ValueError("schema mismatch")

        with pytest.raises(ValueError, match="schema mismatch"):
            await ainvoke_with_rate_limit_retry(_Model(), [], config=None)
        assert _Model.calls == 1


class TestAinvokeWithRateLimitRetryGivesUp:
    @pytest.mark.asyncio
    async def test_raises_provider_rate_limit_after_total_cap(self) -> None:
        """If sustained rate-limits exceed TOTAL_WAIT_CAP_SECONDS, raise."""
        import anthropic

        # Force each retry to want the full single-wait cap so we hit total cap fast.
        response = _build_httpx_response(429, {"retry-after": str(SINGLE_WAIT_CAP_SECONDS)})

        class _Model:
            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                raise anthropic.RateLimitError("rate", response=response, body=None)

        # Patch asyncio.sleep so the test runs fast.
        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        with (
            patch("questfoundry.providers.rate_limit.asyncio.sleep", _fake_sleep),
            pytest.raises(ProviderRateLimitError),
        ):
            await ainvoke_with_rate_limit_retry(_Model(), [], config=None)

        # Some sleeps should have happened before giving up.
        assert sum(slept) <= TOTAL_WAIT_CAP_SECONDS + SINGLE_WAIT_CAP_SECONDS
        assert len(slept) >= 1


class TestBudgetCapDominatesIterationCap:
    @pytest.mark.asyncio
    async def test_short_retry_after_uses_full_time_budget(self) -> None:
        """A provider sending ``Retry-After: 1`` should be retried until the
        full ``TOTAL_WAIT_CAP_SECONDS`` budget is spent, not capped at the
        iteration ceiling.

        Regression for the iteration-cap-vs-budget interaction: an earlier
        ``_MAX_BACKOFF_ATTEMPTS`` of 12 silently bounded short-hint retries to
        ~12 seconds even though the time budget was 300s.
        """
        import anthropic

        response = _build_httpx_response(429, {"retry-after": "1"})

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                raise anthropic.RateLimitError("rate", response=response, body=None)

        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        with (
            patch("questfoundry.providers.rate_limit.asyncio.sleep", _fake_sleep),
            pytest.raises(ProviderRateLimitError),
        ):
            await ainvoke_with_rate_limit_retry(_Model(), [], config=None)

        # With ``Retry-After: 1`` and ±10% jitter, expect roughly TOTAL_WAIT_CAP / 1
        # iterations before the helper gives up — far more than the historical
        # ceiling of 12.
        assert _Model.calls >= 50, (
            f"short Retry-After should let the helper retry many times before the "
            f"time budget is spent — got only {_Model.calls} attempts"
        )
        assert sum(slept) <= TOTAL_WAIT_CAP_SECONDS + SINGLE_WAIT_CAP_SECONDS


class TestBackoffPolicy:
    @pytest.mark.asyncio
    async def test_uses_retry_after_header_when_present(self) -> None:
        """When the response has Retry-After, the helper sleeps approximately that long."""
        import anthropic

        response = _build_httpx_response(429, {"retry-after": "3"})

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                if _Model.calls == 1:
                    raise anthropic.RateLimitError("rate", response=response, body=None)
                return "ok"

        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        with patch("questfoundry.providers.rate_limit.asyncio.sleep", _fake_sleep):
            await ainvoke_with_rate_limit_retry(_Model(), [], config=None)

        # Hint was 3s; with ±10% jitter, expect ~2.7..3.3s
        assert slept and 2.5 <= slept[0] <= 3.5

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_no_retry_after(self) -> None:
        """When no Retry-After header, helper uses DEFAULT_INITIAL_WAIT_SECONDS."""
        import anthropic

        response = _build_httpx_response(429, {})  # no retry-after

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                if _Model.calls == 1:
                    raise anthropic.RateLimitError("rate", response=response, body=None)
                return "ok"

        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        with patch("questfoundry.providers.rate_limit.asyncio.sleep", _fake_sleep):
            await ainvoke_with_rate_limit_retry(_Model(), [], config=None)

        assert slept
        # Default with ±10% jitter
        lower = DEFAULT_INITIAL_WAIT_SECONDS * 0.85
        upper = DEFAULT_INITIAL_WAIT_SECONDS * 1.15
        assert lower <= slept[0] <= upper

    @pytest.mark.asyncio
    async def test_single_sleep_capped(self) -> None:
        """Even when Retry-After says 600s, the helper caps a single sleep."""
        import anthropic

        response = _build_httpx_response(429, {"retry-after": "600"})

        class _Model:
            calls = 0

            async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> str:
                _Model.calls += 1
                if _Model.calls == 1:
                    raise anthropic.RateLimitError("rate", response=response, body=None)
                return "ok"

        slept: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        with patch("questfoundry.providers.rate_limit.asyncio.sleep", _fake_sleep):
            await ainvoke_with_rate_limit_retry(_Model(), [], config=None)

        assert slept and slept[0] <= SINGLE_WAIT_CAP_SECONDS * 1.15


class TestProviderRateLimitErrorShape:
    """ProviderRateLimitError carries provider + retry_after_seconds (issue #1581)."""

    def test_has_retry_after_attribute(self) -> None:
        err = ProviderRateLimitError("anthropic", "throttled", retry_after_seconds=7.0)
        assert err.provider == "anthropic"
        assert err.retry_after_seconds == 7.0

    def test_retry_after_optional(self) -> None:
        err = ProviderRateLimitError("openai", "throttled")
        assert err.provider == "openai"
        assert err.retry_after_seconds is None

    def test_message_format(self) -> None:
        err = ProviderRateLimitError("anthropic", "throttled", retry_after_seconds=2.0)
        assert "anthropic" in str(err)
        assert "throttled" in str(err)


def test_module_constants_have_sane_values() -> None:
    """Sanity: caps and defaults are positive and ordered correctly."""
    assert 0 < DEFAULT_INITIAL_WAIT_SECONDS < SINGLE_WAIT_CAP_SECONDS
    assert SINGLE_WAIT_CAP_SECONDS <= TOTAL_WAIT_CAP_SECONDS


# -----------------------------------------------------------------------------
# Integration: rate-limit error does NOT consume the semantic-retry budget,
# and does NOT append a schema-repair HumanMessage.  Issue #1581 acceptance
# criteria: "no `serialize_error` events with `attempt>=N` in logs are caused
# by 429s — only by genuine validation/parse failures".
# -----------------------------------------------------------------------------


class TestSerializeRetryLoopHonoursRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_does_not_decrement_semantic_budget(self) -> None:
        """A 429 followed by a valid result returns on attempt 1 (no semantic
        retry consumed) — and the retry helper is what handled the 429."""
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import BaseModel, Field

        from questfoundry.agents.serialize import serialize_to_artifact

        class _Schema(BaseModel):
            title: str = Field(min_length=1)
            count: int = Field(ge=1)

        import anthropic

        response = _build_httpx_response(429, {"retry-after": "0.0"})
        rate_limit_exc = anthropic.RateLimitError("rate", response=response, body=None)

        mock_model = MagicMock()
        # First call: 429.  Second call: valid result.  If the retry loop
        # treated the 429 as a semantic error, max_retries would have to be >= 2
        # to succeed.  We pass max_retries=1 to prove the rate-limit retry
        # didn't burn that single semantic slot.
        mock_invoke = AsyncMock(side_effect=[rate_limit_exc, {"title": "Valid", "count": 5}])
        mock_model.with_structured_output.return_value.ainvoke = mock_invoke

        async def _instant_sleep(_seconds: float) -> None:
            return None

        with patch("questfoundry.providers.rate_limit.asyncio.sleep", _instant_sleep):
            artifact, _tokens, attempts_made = await serialize_to_artifact(
                mock_model,
                "A test brief",
                _Schema,
                max_retries=1,
            )

        assert artifact.title == "Valid"
        # Critical: only one *semantic* attempt happened.
        assert attempts_made == 1
        # Two underlying ainvoke calls occurred (one 429, one success).
        assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_grow_rate_limit_does_not_decrement_semantic_budget(self) -> None:
        """A 429 in GROW's _grow_llm_call must not consume the semantic-retry slot.

        GROW has a unique ``continue`` branch (semantic-validator hits >50%
        phantom IDs) that re-runs the loop body — the rate-limit retry must
        not interact with this counter.
        """
        from unittest.mock import AsyncMock, MagicMock

        import anthropic

        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag
        from questfoundry.pipeline.stages.grow import GrowStage

        response = _build_httpx_response(429, {"retry-after": "0.0"})
        rate_limit_exc = anthropic.RateLimitError("rate", response=response, body=None)

        valid_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    beat_id="beat::a",
                    scene_type="scene",
                    narrative_function="introduce",
                    exit_mood="tense anticipation",
                )
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=[rate_limit_exc, valid_output])

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        stage = GrowStage()

        async def _instant_sleep(_seconds: float) -> None:
            return None

        with patch("questfoundry.providers.rate_limit.asyncio.sleep", _instant_sleep):
            _result, llm_calls, _tokens = await stage._grow_llm_call(
                model=mock_model,
                template_name="grow_phase4a_scene_types",
                context={
                    "beat_summaries": "test",
                    "valid_beat_ids": "beat::a",
                    "beat_count": "1",
                },
                output_schema=Phase4aOutput,
                max_retries=1,
            )

        # Only one *semantic* attempt happened (max_retries=1) — the 429 retry
        # was absorbed by the rate-limit helper, not by the schema-repair loop.
        assert llm_calls == 1
        assert mock_structured.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_append_humanmessage(self) -> None:
        """A 429 retry must not append a schema-repair HumanMessage.

        The schema-repair feedback is only appropriate for semantic failures.
        Appending it on rate-limit grows the prompt and worsens the next call's
        rate-limit pressure — exactly the failure mode in issue #1581.

        Snapshot the messages list inside the side_effect because the
        production code passes the same list object to every ainvoke call —
        ``call_args_list`` would alias the post-mutation list otherwise.
        """
        from unittest.mock import MagicMock

        from langchain_core.messages import HumanMessage
        from pydantic import BaseModel, Field

        from questfoundry.agents.serialize import serialize_to_artifact

        class _Schema(BaseModel):
            title: str = Field(min_length=1)
            count: int = Field(ge=1)

        import anthropic

        response = _build_httpx_response(429, {"retry-after": "0.0"})
        rate_limit_exc = anthropic.RateLimitError("rate", response=response, body=None)

        snapshots: list[list[Any]] = []

        async def _ainvoke(messages: list[Any], config: Any = None) -> dict[str, Any]:
            del config  # unused — only the messages snapshot matters here
            snapshots.append(list(messages))
            if len(snapshots) == 1:
                raise rate_limit_exc
            return {"title": "Valid", "count": 5}

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = _ainvoke

        async def _instant_sleep(_seconds: float) -> None:
            return None

        with patch("questfoundry.providers.rate_limit.asyncio.sleep", _instant_sleep):
            await serialize_to_artifact(
                mock_model,
                "A test brief",
                _Schema,
                max_retries=1,
            )

        # Two calls happened (one 429, one success). The HumanMessage count in
        # the second call must equal the first — i.e. no schema-repair message
        # was appended between them.
        assert len(snapshots) == 2
        first_humans = sum(1 for m in snapshots[0] if isinstance(m, HumanMessage))
        second_humans = sum(1 for m in snapshots[1] if isinstance(m, HumanMessage))
        assert first_humans == second_humans, (
            "Schema-repair HumanMessage must NOT be appended on rate-limit retry "
            f"(first={first_humans}, second={second_humans})"
        )
