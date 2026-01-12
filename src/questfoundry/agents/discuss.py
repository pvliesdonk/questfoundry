"""Discuss phase agent for creative exploration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from questfoundry.agents.prompts import get_discuss_prompt
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

log = get_logger(__name__)

# Type aliases for callbacks
#: Async function to get user input; returns None or empty string to exit
UserInputFn = Callable[[], Awaitable[str | None]]
#: Callback when assistant responds; receives the response content
AssistantMessageFn = Callable[[str], None]
#: Callback for LLM start/end events; receives the phase name (e.g., "discuss")
LLMCallbackFn = Callable[[str], None]


def create_discuss_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
) -> Any:  # Returns CompiledStateGraph but avoid import issues
    """Create a Discuss phase agent.

    Uses LangChain's create_agent() (v1.0 API) to create an agent that
    can discuss creative vision with the user while using research tools.

    Args:
        model: Chat model to use
        tools: Research tools available to the agent

    Returns:
        Compiled agent graph ready for invocation
    """
    from langchain.agents import create_agent

    # Use consistent logic for checking if tools are available
    has_tools = bool(tools)

    system_prompt = get_discuss_prompt(
        research_tools_available=has_tools,
    )

    log.info("discuss_agent_created", tool_count=len(tools) if tools else 0)

    return create_agent(
        model=model,
        tools=tools if has_tools else None,
        system_prompt=system_prompt,
    )


async def run_discuss_phase(
    model: BaseChatModel,
    tools: list[BaseTool],
    user_prompt: str,
    max_iterations: int = 25,
    *,
    interactive: bool = False,
    user_input_fn: UserInputFn | None = None,
    on_assistant_message: AssistantMessageFn | None = None,
    on_llm_start: LLMCallbackFn | None = None,
    on_llm_end: LLMCallbackFn | None = None,
) -> tuple[list[BaseMessage], int, int]:
    """Run the Discuss phase to completion.

    Creates a Discuss agent and runs it with the user's initial prompt.
    The agent will use research tools as needed and return when complete.

    In interactive mode, allows multi-turn conversation until user types
    '/done' or provides empty input.

    Args:
        model: Chat model to use
        tools: Research tools
        user_prompt: User's initial story idea
        max_iterations: Maximum agent iterations per turn (default 25)
        interactive: Whether to enable interactive multi-turn mode
        user_input_fn: Async function to get user input (required for interactive)
        on_assistant_message: Callback when assistant responds
        on_llm_start: Callback when LLM call starts
        on_llm_end: Callback when LLM call ends

    Returns:
        Tuple of (messages, llm_call_count, total_tokens)
    """
    agent = create_discuss_agent(model, tools)

    log.info(
        "discuss_phase_started",
        user_prompt_length=len(user_prompt),
        interactive=interactive,
    )

    # Track all messages and metrics across turns
    all_messages: list[BaseMessage] = []
    total_llm_calls = 0
    total_tokens = 0

    # Initial message from user
    current_messages: list[BaseMessage] = [HumanMessage(content=user_prompt)]

    # Validate interactive mode requirements upfront
    if interactive and user_input_fn is None:
        log.error("interactive_mode_no_input_fn", stage="discuss")
        raise ValueError(
            "interactive=True requires user_input_fn callback. "
            "Pass user_input_fn parameter or set interactive=False."
        )

    while True:
        # Signal LLM start
        if on_llm_start:
            on_llm_start("discuss")

        # Run agent for this turn
        result = await agent.ainvoke(
            {"messages": current_messages},
            config={"recursion_limit": max_iterations},
        )

        # Signal LLM end
        if on_llm_end:
            on_llm_end("discuss")

        # The agent returns the full conversation history.
        # We need to process only the messages added in this turn.
        full_history: list[BaseMessage] = result.get("messages", [])

        # Determine new messages: check if agent returned full history including our input,
        # or just the response. Real agents include input; test mocks may not.
        if (
            len(full_history) > len(current_messages)
            and len(current_messages) > 0
            and full_history[0].content == current_messages[0].content
        ):
            # Agent returned full history - slice off our input to get new messages
            new_messages = full_history[len(current_messages) :]
        else:
            # Agent returned only responses (or test mock) - treat all as new
            new_messages = full_history

        # Extract metrics and find the last assistant message from new messages only
        last_ai_content = ""
        for msg in new_messages:
            if isinstance(msg, AIMessage):
                total_llm_calls += 1
                last_ai_content = str(msg.content)
                # Extract tokens
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    tokens = msg.usage_metadata.get("total_tokens")
                    total_tokens += tokens if tokens is not None else 0
                elif hasattr(msg, "response_metadata") and msg.response_metadata:
                    metadata = msg.response_metadata
                    if "token_usage" in metadata:
                        usage = metadata["token_usage"]
                        tokens = usage.get("total_tokens")
                        total_tokens += tokens if tokens is not None else 0

        # Update message history for the next turn
        all_messages = full_history

        # Display assistant response
        if on_assistant_message and last_ai_content:
            on_assistant_message(last_ai_content)

        # If not interactive, we're done after one turn
        if not interactive:
            break

        # Interactive mode: get next user input
        assert user_input_fn is not None  # Validated at line 112
        user_input = await user_input_fn()

        # Check for exit conditions
        if user_input is None or user_input.strip() == "":
            log.debug("interactive_exit", reason="empty_input")
            break

        if user_input.strip().lower() in ("/done", "/quit", "/exit"):
            log.debug("interactive_exit", reason="user_command")
            break

        # Continue conversation with user's new input
        current_messages = [*all_messages, HumanMessage(content=user_input)]

    log.info(
        "discuss_phase_completed",
        message_count=len(all_messages),
        llm_calls=total_llm_calls,
        total_tokens=total_tokens,
        interactive=interactive,
    )

    return all_messages, total_llm_calls, total_tokens
