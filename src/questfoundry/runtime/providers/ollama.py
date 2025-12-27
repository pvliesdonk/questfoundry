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
                # Include tool_calls on AIMessage if present
                if msg.tool_calls:
                    tool_calls = [
                        {"id": tc.id, "name": tc.name, "args": tc.arguments}
                        for tc in msg.tool_calls
                    ]
                    langchain_messages.append(AIMessage(content=msg.content, tool_calls=tool_calls))
                else:
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

        # Build langchain ChatOllama with optional model-specific options
        ollama_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": self._host,
            "temperature": options.temperature,
            "num_predict": options.max_tokens,
            "stop": options.stop_sequences if options.stop_sequences else None,
        }
        # Add num_ctx if specified in model_options
        if num_ctx := options.model_options.get("num_ctx"):
            ollama_kwargs["num_ctx"] = num_ctx

        llm: Any = ChatOllama(**ollama_kwargs)

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

        # Build langchain ChatOllama with optional model-specific options
        ollama_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": self._host,
            "temperature": options.temperature,
            "num_predict": options.max_tokens,
            "stop": options.stop_sequences if options.stop_sequences else None,
        }
        # Add num_ctx if specified in model_options
        if num_ctx := options.model_options.get("num_ctx"):
            ollama_kwargs["num_ctx"] = num_ctx

        llm: Any = ChatOllama(**ollama_kwargs)

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
            # Track tool call chunks by index to accumulate incrementally
            # Key: index, Value: {"id": str, "name": str, "args_json": str}
            tool_call_builders: dict[int, dict[str, str]] = {}

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

                # Accumulate tool call chunks (args come as JSON string fragments)
                tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
                if tool_call_chunks:
                    for tcc in tool_call_chunks:
                        idx = tcc.get("index", 0)
                        if idx not in tool_call_builders:
                            tool_call_builders[idx] = {"id": "", "name": "", "args_json": ""}

                        builder = tool_call_builders[idx]
                        if tcc.get("id"):
                            builder["id"] = tcc["id"]
                        if tcc.get("name"):
                            builder["name"] = tcc["name"]
                        if tcc.get("args"):
                            builder["args_json"] += tcc["args"]

                yield StreamChunk(
                    content=content,
                    done=False,
                )

            # Parse accumulated tool calls
            import json

            final_tool_calls = []
            for _idx, builder in tool_call_builders.items():
                if builder["name"] and builder["id"]:
                    # Parse the accumulated JSON args
                    try:
                        args = json.loads(builder["args_json"]) if builder["args_json"] else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool call args: {builder['args_json']}")
                        args = {}

                    final_tool_calls.append(
                        ToolCallRequest(
                            id=builder["id"],
                            name=builder["name"],
                            arguments=args,
                        )
                    )

            # Final chunk with done=True, usage stats, and parsed tool calls
            yield StreamChunk(
                content="",
                done=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tool_calls=final_tool_calls if final_tool_calls else None,
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

    async def get_context_size(self, model: str) -> int | None:
        """
        Get context window size for an Ollama model.

        Uses the /api/show endpoint to get model details.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self._host}/api/show",
                json={"name": model},
            )
            if response.status_code == 200:
                data = response.json()
                # Context length is in model_info.context_length or
                # can be parsed from parameters string
                model_info = data.get("model_info", {})

                # Try model_info first (structured data)
                for key, value in model_info.items():
                    if "context_length" in key.lower() and isinstance(value, int):
                        return value

                # Fallback: parse from parameters string
                # Format like "num_ctx 8192" or similar
                params = data.get("parameters", "")
                if "num_ctx" in params:
                    for line in params.split("\n"):
                        if "num_ctx" in line:
                            parts = line.split()
                            for part in parts:
                                if part.isdigit():
                                    return int(part)

                logger.debug(f"Could not find context size in Ollama model info for {model}")
                return None
            else:
                logger.debug(f"Ollama /api/show returned {response.status_code} for {model}")
                return None
        except Exception as e:
            logger.debug(f"Failed to get context size for {model}: {e}")
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
