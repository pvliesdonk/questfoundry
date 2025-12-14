"""
Agent runtime for executing agent activations.

The AgentRuntime handles the full lifecycle of an agent activation:
1. Load agent and build context
2. Build system prompt
3. Validate context size
4. Execute LLM call (streaming or non-streaming)
5. Track turn in session
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from questfoundry.runtime.agent.context import AgentContext, ContextBuilder
from questfoundry.runtime.agent.prompt import PromptBuilder, build_prompt
from questfoundry.runtime.providers import (
    ContextOverflowError,
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    StreamChunk,
)
from questfoundry.runtime.session import Session, TokenUsage, Turn

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, Studio


@dataclass
class ActivationResult:
    """Result of an agent activation."""

    content: str
    agent_id: str
    turn: Turn
    usage: TokenUsage | None = None


class AgentRuntime:
    """
    Runtime for executing agent activations.

    Handles context building, prompt construction, LLM invocation,
    and session/turn management.
    """

    def __init__(
        self,
        provider: LLMProvider,
        studio: Studio,
        domain_path: Path | None = None,
        model: str = "qwen3:8b",
        context_limit: int | None = None,
    ):
        """
        Initialize agent runtime.

        Args:
            provider: LLM provider to use
            studio: Loaded studio definition
            domain_path: Path to domain directory (for knowledge loading)
            model: Model to use for LLM calls
            context_limit: Maximum context size (tokens), raises error if exceeded
        """
        self._provider = provider
        self._studio = studio
        self._domain_path = domain_path
        self._model = model
        self._context_limit = context_limit

        self._context_builder = ContextBuilder(domain_path=domain_path)
        self._prompt_builder = PromptBuilder()

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        for agent in self._studio.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_entry_agent(self) -> Agent | None:
        """Get the first entry agent."""
        for agent in self._studio.agents:
            if agent.is_entry_agent:
                return agent
        return None

    def build_context(self, agent: Agent) -> AgentContext:
        """Build context for an agent."""
        return self._context_builder.build(agent, self._studio)

    def build_messages(
        self,
        agent: Agent,
        user_input: str,
        context: AgentContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[LLMMessage]:
        """
        Build messages for LLM invocation.

        Args:
            agent: The agent to activate
            user_input: User's input message
            context: Pre-built context (built if not provided)
            history: Conversation history [{role, content}, ...]

        Returns:
            List of LLMMessage for the provider
        """
        if context is None:
            context = self.build_context(agent)

        # Build system prompt
        prompt = build_prompt(
            agent=agent,
            constitution_text=context.constitution_text,
            must_know_entries=context.must_know_entries,
            role_specific_menu=context.role_specific_menu,
        )

        messages: list[LLMMessage] = []

        # System message
        messages.append(LLMMessage(role="system", content=prompt.text))

        # History
        if history:
            for h in history:
                messages.append(LLMMessage(role=h["role"], content=h["content"]))

        # User input
        messages.append(LLMMessage(role="user", content=user_input))

        return messages

    def estimate_tokens(self, messages: list[LLMMessage]) -> int:
        """Estimate token count for messages."""
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4  # Rough estimate

    def validate_context_size(self, messages: list[LLMMessage]) -> None:
        """
        Validate that messages fit within context limit.

        Raises:
            ContextOverflowError: If messages exceed context limit
        """
        if self._context_limit is None:
            return

        estimated = self.estimate_tokens(messages)
        if estimated > self._context_limit:
            raise ContextOverflowError(
                f"Prompt ({estimated} tokens) exceeds model context "
                f"({self._context_limit} tokens). Reduce knowledge or use larger model."
            )

    async def activate(
        self,
        agent: Agent,
        user_input: str,
        session: Session,
        options: InvokeOptions | None = None,
    ) -> ActivationResult:
        """
        Activate an agent (non-streaming).

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options

        Returns:
            ActivationResult with response content and turn

        Raises:
            ContextOverflowError: If prompt exceeds context limit
            ProviderError: If LLM call fails
        """
        # Start turn
        turn = session.start_turn(agent.id, user_input)

        try:
            # Build context and messages
            context = self.build_context(agent)
            history = session.get_history()[:-2] if len(session.turns) > 1 else None
            messages = self.build_messages(agent, user_input, context, history)

            # Validate context size
            self.validate_context_size(messages)

            # Invoke LLM
            response = await self._provider.invoke(messages, self._model, options)

            # Complete turn
            usage = TokenUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            )
            session.complete_turn(turn, response.content, usage)

            return ActivationResult(
                content=response.content,
                agent_id=agent.id,
                turn=turn,
                usage=usage,
            )

        except Exception as e:
            session.error_turn(turn, str(e))
            raise

    async def activate_streaming(
        self,
        agent: Agent,
        user_input: str,
        session: Session,
        options: InvokeOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Activate an agent with streaming response.

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options

        Yields:
            StreamChunk with content

        Raises:
            ContextOverflowError: If prompt exceeds context limit
            ProviderError: If LLM call fails
        """
        # Start turn
        turn = session.start_turn(agent.id, user_input)

        try:
            # Build context and messages
            context = self.build_context(agent)
            history = session.get_history()[:-2] if len(session.turns) > 1 else None
            messages = self.build_messages(agent, user_input, context, history)

            # Validate context size
            self.validate_context_size(messages)

            # Collect response for turn completion
            full_content = ""
            final_usage: TokenUsage | None = None

            # Stream from provider
            async for chunk in self._provider.stream(messages, self._model, options):
                full_content += chunk.content

                if chunk.done:
                    final_usage = TokenUsage(
                        prompt_tokens=chunk.prompt_tokens,
                        completion_tokens=chunk.completion_tokens,
                        total_tokens=chunk.total_tokens,
                    )

                yield chunk

            # Complete turn after streaming finishes
            session.complete_turn(turn, full_content, final_usage)

        except Exception as e:
            session.error_turn(turn, str(e))
            raise


async def activate_agent(
    agent_id: str,
    user_input: str,
    runtime: AgentRuntime,
    session: Session,
    streaming: bool = True,
    options: InvokeOptions | None = None,
) -> ActivationResult | AsyncIterator[StreamChunk]:
    """
    Convenience function to activate an agent.

    Args:
        agent_id: ID of the agent to activate
        user_input: User's input message
        runtime: The agent runtime
        session: Session to track the turn
        streaming: Whether to use streaming
        options: LLM invocation options

    Returns:
        ActivationResult (non-streaming) or AsyncIterator[StreamChunk] (streaming)

    Raises:
        ValueError: If agent not found
        ContextOverflowError: If prompt exceeds context limit
    """
    agent = runtime.get_agent(agent_id)
    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    if streaming:
        return runtime.activate_streaming(agent, user_input, session, options)
    else:
        return await runtime.activate(agent, user_input, session, options)
