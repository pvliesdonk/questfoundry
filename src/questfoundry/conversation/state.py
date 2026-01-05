"""Conversation state tracking.

This module provides the ConversationState dataclass for tracking
the state of a conversation during interactive stage execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.providers.base import Message


@dataclass
class ConversationState:
    """Tracks state during a conversation loop.

    Maintains the message history, turn count, phase, and resource usage
    metrics for a conversation session.

    The phase tracks the current stage of the 3-phase conversation pattern:
    - "discuss": Initial discussion with research tools
    - "summarize": Generating summary with no tools
    - "serialize": Converting to structured output with finalization tool

    Attributes:
        messages: List of all messages in the conversation.
        turn_count: Number of completed conversation turns in discuss phase.
        llm_calls: Total LLM API calls made.
        tokens_used: Total tokens consumed across all calls.
        phase: Current phase ("discuss", "summarize", or "serialize").

    Example:
        >>> state = ConversationState(phase="discuss")
        >>> state.messages.append({"role": "user", "content": "Hello"})
        >>> state.turn_count += 1
        >>> state.llm_calls += 1
        >>> state.tokens_used += 150
        >>> state.phase = "summarize"
    """

    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    llm_calls: int = 0
    tokens_used: int = 0
    phase: str = "discuss"

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message to the conversation."""
        self.messages.append(
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            }
        )
