"""Context variables for interactive tool callbacks.

This module provides async-safe storage for interactive callbacks that tools
can access during execution. Uses contextvars for proper async isolation.

The callbacks are set before agent invocation and cleared after, allowing
interactive tools (like present_options) to display UI and collect user input
without requiring changes to the Tool protocol.

Usage in discuss.py:
    from questfoundry.tools.interactive_context import set_interactive_callbacks

    async def run_discuss_phase(...):
        if interactive:
            set_interactive_callbacks(user_input_fn, on_assistant_message)
        try:
            # ... agent invocation ...
        finally:
            clear_interactive_callbacks()

Usage in tools:
    from questfoundry.tools.interactive_context import get_interactive_callbacks

    class PresentOptionsTool:
        def execute(self, arguments):
            callbacks = get_interactive_callbacks()
            if callbacks is None:
                return '{"result": "skipped", ...}'
            # Use callbacks to display UI and get input
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

# Type aliases matching discuss.py
UserInputFn = Callable[[], Awaitable[str | None]]
DisplayFn = Callable[[str], None]


@dataclass
class InteractiveCallbacks:
    """Container for interactive mode callbacks.

    Attributes:
        user_input_fn: Async function to get user input.
        display_fn: Function to display formatted content to user.
        event_loop: The event loop to use for async operations from sync context.
    """

    user_input_fn: UserInputFn
    display_fn: DisplayFn
    event_loop: asyncio.AbstractEventLoop


# Context variable for async-safe callback storage
_interactive_callbacks: contextvars.ContextVar[InteractiveCallbacks | None] = (
    contextvars.ContextVar("interactive_callbacks", default=None)
)


def set_interactive_callbacks(
    user_input_fn: UserInputFn,
    display_fn: DisplayFn,
) -> None:
    """Set callbacks for current async context.

    Call this before agent invocation when in interactive mode.
    The callbacks will be available to any tools that need them.

    Also captures the current event loop so that sync tools can schedule
    async operations (like user input) even when running in a thread pool.

    Args:
        user_input_fn: Async function that prompts user and returns input.
        display_fn: Function to display formatted content (e.g., rich panel).
    """
    loop = asyncio.get_running_loop()
    _interactive_callbacks.set(InteractiveCallbacks(user_input_fn, display_fn, loop))


def get_interactive_callbacks() -> InteractiveCallbacks | None:
    """Get callbacks for current async context.

    Returns:
        InteractiveCallbacks if in interactive mode, None otherwise.
    """
    return _interactive_callbacks.get()


def clear_interactive_callbacks() -> None:
    """Clear callbacks after agent invocation completes."""
    _interactive_callbacks.set(None)
