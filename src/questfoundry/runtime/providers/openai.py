"""
OpenAI LLM Provider implementation.

Uses langchain-openai for communication with OpenAI API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from questfoundry.runtime.providers.base import (
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ProviderConfigError,
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider using langchain-openai.

    Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional custom base URL (for Azure or compatible APIs)
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url

        if not self._api_key:
            raise ProviderConfigError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return "openai"

    def _get_llm(self, model: str, options: InvokeOptions):  # type: ignore[no-untyped-def]
        """Create a ChatOpenAI instance with the given settings."""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=self._api_key,
            temperature=options.temperature,
            timeout=options.timeout_seconds,
            max_tokens=options.max_tokens,
            stop=options.stop_sequences if options.stop_sequences else None,
            base_url=self._base_url,
        )

    def _convert_messages(self, messages: list[LLMMessage]):  # type: ignore[no-untyped-def]
        """Convert LLMMessage to langchain format."""
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

        langchain_messages: list[BaseMessage] = []
        for msg in messages:
            if msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        return langchain_messages

    async def invoke(
        self,
        messages: list[LLMMessage],
        model: str,
        options: InvokeOptions | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """
        Send messages to OpenAI and get a response.

        Uses langchain-openai for the actual invocation.
        Note: tools parameter accepted but not yet implemented.
        """
        # TODO: Implement tool support for OpenAI
        _ = tools  # Acknowledge but not yet used
        options = options or InvokeOptions()

        llm = self._get_llm(model, options)
        langchain_messages = self._convert_messages(messages)

        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(langchain_messages),
                timeout=options.timeout_seconds,
            )
        except TimeoutError as e:
            raise ProviderError(f"OpenAI request timed out after {options.timeout_seconds}s") from e
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ProviderUnavailableError(f"OpenAI authentication failed: {e}") from e
            raise ProviderError(f"OpenAI invocation failed: {e}") from e

        duration_ms = (time.time() - start_time) * 1000

        # Extract token usage
        usage_metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if usage_metadata:
            prompt_tokens = usage_metadata.get("input_tokens")
            completion_tokens = usage_metadata.get("output_tokens")
            total_tokens = usage_metadata.get("total_tokens")

        # Extract content
        content = response.content
        if isinstance(content, list):
            content = str(content[0]) if content else ""

        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            raw=response,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        options: InvokeOptions | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response chunks from OpenAI.

        Uses langchain-openai's astream for streaming.
        Note: tools parameter accepted but not yet implemented.
        """
        # TODO: Implement tool support for OpenAI
        _ = tools  # Acknowledge but not yet used
        options = options or InvokeOptions()

        llm = self._get_llm(model, options)
        langchain_messages = self._convert_messages(messages)

        try:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None

            async for chunk in llm.astream(langchain_messages):
                content = chunk.content
                if isinstance(content, list):
                    content = str(content[0]) if content else ""

                # Check for usage metadata on final chunk
                usage_metadata = getattr(chunk, "usage_metadata", None)
                if usage_metadata:
                    prompt_tokens = usage_metadata.get("input_tokens")
                    completion_tokens = usage_metadata.get("output_tokens")
                    total_tokens = usage_metadata.get("total_tokens")

                yield StreamChunk(
                    content=content,
                    done=False,
                )

            # Final chunk with done=True and usage stats
            yield StreamChunk(
                content="",
                done=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            raise ProviderError(f"OpenAI streaming failed: {e}") from e

    async def check_availability(self) -> bool:
        """Check if OpenAI API is reachable with valid credentials."""
        try:
            # Use a minimal models list call to verify connectivity
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
            await client.models.list()
            return True
        except Exception as e:
            logger.debug(f"OpenAI availability check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List models available from OpenAI."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
            models = await client.models.list()
            return [model.id for model in models.data if "gpt" in model.id.lower()]
        except Exception as e:
            logger.debug(f"Failed to list OpenAI models: {e}")
            return []

    async def close(self) -> None:
        """No persistent connections to close for OpenAI."""
        pass
