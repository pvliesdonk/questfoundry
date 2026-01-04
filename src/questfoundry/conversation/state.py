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

    Maintains the message history, turn count, and resource usage
    metrics for a conversation session.

    Attributes:
        messages: List of all messages in the conversation.
        turn_count: Number of completed conversation turns.
        llm_calls: Total LLM API calls made.
        tokens_used: Total tokens consumed across all calls.

    Example:
        >>> state = ConversationState()
        >>> state.messages.append({"role": "user", "content": "Hello"})
        >>> state.turn_count += 1
        >>> state.llm_calls += 1
        >>> state.tokens_used += 150
    """

    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    llm_calls: int = 0
    tokens_used: int = 0

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
