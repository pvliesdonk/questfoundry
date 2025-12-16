"""
Google (Gemini) LLM Provider implementation.

Uses langchain-google-genai for communication with Google's Gemini API.
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


class GoogleProvider(LLMProvider):
    """
    Google Gemini provider using langchain-google-genai.

    Requires GOOGLE_API_KEY environment variable or explicit api_key.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Google provider.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not self._api_key:
            raise ProviderConfigError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    @property
    def name(self) -> str:
        return "google"

    def _get_llm(self, model: str, options: InvokeOptions):  # type: ignore[no-untyped-def]
        """Create a ChatGoogleGenerativeAI instance with the given settings."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self._api_key,
            temperature=options.temperature,
            timeout=options.timeout_seconds,
            max_output_tokens=options.max_tokens,
            stop=options.stop_sequences if options.stop_sequences else None,
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
        callbacks: list[Any] | None = None,
    ) -> LLMResponse:
        """
        Send messages to Google Gemini and get a response.

        Uses langchain-google-genai for the actual invocation.
        Note: tools parameter accepted but not yet implemented.
        """
        # TODO: Implement tool support for Google
        _ = tools  # Acknowledge but not yet used
        _ = callbacks  # Acknowledge but not yet used
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
            raise ProviderError(f"Google request timed out after {options.timeout_seconds}s") from e
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ProviderUnavailableError(f"Google authentication failed: {e}") from e
            raise ProviderError(f"Google invocation failed: {e}") from e

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
        callbacks: list[Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response chunks from Google Gemini.

        Uses langchain-google-genai's astream for streaming.
        Note: tools parameter accepted but not yet implemented.
        """
        # TODO: Implement tool support for Google
        _ = tools  # Acknowledge but not yet used
        _ = callbacks  # Acknowledge but not yet used
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
            raise ProviderError(f"Google streaming failed: {e}") from e

    async def check_availability(self) -> bool:
        """Check if Google API is reachable with valid credentials."""
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]

            genai.configure(api_key=self._api_key)
            # List models to verify API key works
            list(genai.list_models())
            return True
        except Exception as e:
            logger.debug(f"Google availability check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List models available from Google."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            models = list(genai.list_models())
            return [
                m.name.replace("models/", "")
                for m in models
                if "generateContent" in (m.supported_generation_methods or [])
            ]
        except Exception as e:
            logger.debug(f"Failed to list Google models: {e}")
            return []

    async def close(self) -> None:
        """No persistent connections to close for Google."""
        pass
