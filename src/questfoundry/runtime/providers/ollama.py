"""
Ollama LLM Provider implementation.

Uses langchain-ollama for communication with local Ollama instance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from langchain_ollama import ChatOllama

from questfoundry.runtime.providers.base import (
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama provider using langchain-ollama.

    Connects to a local or remote Ollama instance.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.

        Args:
            host: Ollama server URL (default: http://localhost:11434)
        """
        self._host = host.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def host(self) -> str:
        return self._host

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[Any]:
        """Convert LLMMessages to LangChain message format."""
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        langchain_messages: list[BaseMessage] = []
        for msg in messages:
            if msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == "tool":
                # Tool result message
                langchain_messages.append(
                    ToolMessage(
                        content=msg.content,
                        tool_call_id=msg.tool_call_id or "",
                        name=msg.name,
                    )
                )
        return langchain_messages

    def _parse_tool_calls(self, response: Any) -> list[ToolCallRequest] | None:
        """Parse tool calls from LangChain AIMessage response."""
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return None

        result = []
        for tc in tool_calls:
            result.append(
                ToolCallRequest(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("args", {}),
                )
            )
        return result if result else None

    async def invoke(
        self,
        messages: list[LLMMessage],
        model: str,
        options: InvokeOptions | None = None,
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
    ) -> LLMResponse:
        """
        Send messages to Ollama and get a response.

        Uses langchain-ollama for the actual invocation.
        Supports tool binding when tools are provided.
        """
        options = options or InvokeOptions()

        # Check availability first
        if not await self.check_availability():
            raise ProviderUnavailableError(f"Ollama not available at {self._host}")

        # Build langchain ChatOllama
        llm: Any = ChatOllama(
            model=model,
            base_url=self._host,
            temperature=options.temperature,
            num_predict=options.max_tokens,
            stop=options.stop_sequences if options.stop_sequences else None,
        )

        # Bind tools if provided
        if tools:
            llm = llm.bind_tools(tools)

        # Convert messages to langchain format
        langchain_messages = self._convert_messages(messages)

        # Build config with callbacks for LangSmith tracing
        config = {"callbacks": callbacks} if callbacks else None

        # Invoke with timeout
        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(langchain_messages, config=config),
                timeout=options.timeout_seconds,
            )
        except TimeoutError as e:
            raise ProviderError(f"Ollama request timed out after {options.timeout_seconds}s") from e
        except Exception as e:
            raise ProviderError(f"Ollama invocation failed: {e}") from e

        duration_ms = (time.time() - start_time) * 1000

        # Extract token usage if available
        usage_metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if usage_metadata:
            prompt_tokens = usage_metadata.get("input_tokens")
            completion_tokens = usage_metadata.get("output_tokens")
            total_tokens = usage_metadata.get("total_tokens")

        # Extract content (handle potential list type from langchain)
        content = response.content
        if isinstance(content, list):
            content = str(content[0]) if content else ""

        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response)

        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
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
        Stream response chunks from Ollama.

        Uses langchain-ollama's astream for streaming.
        Tool calls are accumulated and returned in the final chunk.
        """
        options = options or InvokeOptions()

        # Check availability first
        if not await self.check_availability():
            raise ProviderUnavailableError(f"Ollama not available at {self._host}")

        # Build langchain ChatOllama
        llm: Any = ChatOllama(
            model=model,
            base_url=self._host,
            temperature=options.temperature,
            num_predict=options.max_tokens,
            stop=options.stop_sequences if options.stop_sequences else None,
        )

        # Bind tools if provided
        if tools:
            llm = llm.bind_tools(tools)

        # Convert messages to langchain format
        langchain_messages = self._convert_messages(messages)

        # Build config with callbacks for LangSmith tracing
        config = {"callbacks": callbacks} if callbacks else None

        # Stream with timeout
        try:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            accumulated_tool_calls: list[ToolCallRequest] = []

            async for chunk in llm.astream(langchain_messages, config=config):
                # Extract content from chunk
                content = chunk.content
                if isinstance(content, list):
                    content = str(content[0]) if content else ""

                # Check for usage metadata on final chunk
                usage_metadata = getattr(chunk, "usage_metadata", None)
                if usage_metadata:
                    prompt_tokens = usage_metadata.get("input_tokens")
                    completion_tokens = usage_metadata.get("output_tokens")
                    total_tokens = usage_metadata.get("total_tokens")

                # Accumulate tool calls from streaming chunks
                chunk_tool_calls = self._parse_tool_calls(chunk)
                if chunk_tool_calls:
                    accumulated_tool_calls.extend(chunk_tool_calls)

                yield StreamChunk(
                    content=content,
                    done=False,
                )

            # Final chunk with done=True, usage stats, and accumulated tool calls
            yield StreamChunk(
                content="",
                done=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
            )

        except Exception as e:
            raise ProviderError(f"Ollama streaming failed: {e}") from e

    async def check_availability(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._host}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List models available in Ollama."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            logger.debug(f"Failed to list Ollama models: {e}")
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
