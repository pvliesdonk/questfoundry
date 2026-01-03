"""Logging wrapper for LLM providers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.observability import LLMLogger
    from questfoundry.providers import LLMProvider
    from questfoundry.providers.base import LLMResponse, Message
    from questfoundry.tools import ToolDefinition


class LoggingProvider:
    """Wrapper that logs all LLM calls to the LLMLogger.

    Delegates to an underlying provider and logs request/response details
    to the configured LLMLogger.

    Attributes:
        default_model: The model name from the wrapped provider.
    """

    def __init__(
        self,
        provider: LLMProvider,
        logger: LLMLogger,
        stage: str,
    ) -> None:
        """Initialize logging wrapper.

        Args:
            provider: Underlying LLM provider to wrap.
            logger: LLMLogger instance for recording calls.
            stage: Current pipeline stage name for log entries.
        """
        self._provider = provider
        self._logger = logger
        self._stage = stage

    @property
    def default_model(self) -> str:
        """Return the default model from wrapped provider."""
        return self._provider.default_model

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Generate completion and log the call.

        Args:
            messages: List of conversation messages.
            model: Optional model override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Optional tools to bind.
            tool_choice: Tool selection mode.

        Returns:
            LLMResponse from underlying provider.
        """
        start_time = time.perf_counter()
        error_msg: str | None = None

        try:
            response = await self._provider.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as e:
            error_msg = str(e)
            duration = time.perf_counter() - start_time
            # Log the failed call
            entry = self._logger.create_entry(
                stage=self._stage,
                model=model or self.default_model,
                messages=[{**m} for m in messages],  # type: ignore[dict-item]
                content="",
                tokens_used=0,
                finish_reason="error",
                duration_seconds=duration,
                temperature=temperature,
                max_tokens=max_tokens,
                error=error_msg,
            )
            self._logger.log(entry)
            raise

        duration = time.perf_counter() - start_time

        # Log successful call (including tool calls if present)
        entry = self._logger.create_entry(
            stage=self._stage,
            model=response.model,
            messages=[{**m} for m in messages],  # type: ignore[dict-item]
            content=response.content,
            tokens_used=response.tokens_used,
            finish_reason=response.finish_reason,
            duration_seconds=duration,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._logger.log(entry)

        return response

    async def close(self) -> None:
        """Close underlying provider."""
        if hasattr(self._provider, "close"):
            await self._provider.close()

    async def __aenter__(self) -> LoggingProvider:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context."""
        await self.close()
