"""Discuss phase agent for creative exploration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from questfoundry.agents.prompts import get_discuss_prompt
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

log = get_logger(__name__)


def create_discuss_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    user_prompt: str,
) -> Any:  # Returns CompiledStateGraph but avoid import issues
    """Create a Discuss phase agent.

    Uses LangChain's create_agent() (v1.0 API) to create an agent that
    can discuss creative vision with the user while using research tools.

    Args:
        model: Chat model to use
        tools: Research tools available to the agent
        user_prompt: User's initial story idea

    Returns:
        Compiled agent graph ready for invocation
    """
    from langchain.agents import create_agent

    system_prompt = get_discuss_prompt(
        user_prompt=user_prompt,
        research_tools_available=len(tools) > 0,
    )

    log.info("discuss_agent_created", tool_count=len(tools))

    return create_agent(
        model=model,
        tools=tools if tools else None,
        system_prompt=system_prompt,
    )


async def run_discuss_phase(
    model: BaseChatModel,
    tools: list[BaseTool],
    user_prompt: str,
    max_iterations: int = 25,
) -> tuple[list[BaseMessage], int, int]:
    """Run the Discuss phase to completion.

    Creates a Discuss agent and runs it with the user's initial prompt.
    The agent will use research tools as needed and return when complete.

    Args:
        model: Chat model to use
        tools: Research tools
        user_prompt: User's initial story idea
        max_iterations: Maximum agent iterations (default 25)

    Returns:
        Tuple of (messages, llm_call_count, total_tokens)
    """
    agent = create_discuss_agent(model, tools, user_prompt)

    log.info("discuss_phase_started", user_prompt_length=len(user_prompt))

    # Initial message from user - the agent will respond to this
    initial_message = HumanMessage(content=user_prompt)

    result = await agent.ainvoke(
        {"messages": [initial_message]},
        config={"recursion_limit": max_iterations},
    )

    messages: list[BaseMessage] = result.get("messages", [])

    # Extract metrics from response metadata
    # LangChain tracks these in AIMessage.response_metadata
    llm_calls = 0
    total_tokens = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "response_metadata"):
            metadata = msg.response_metadata or {}
            if "token_usage" in metadata:
                usage = metadata["token_usage"]
                total_tokens += usage.get("total_tokens", 0)
                llm_calls += 1
            # Some providers use different metadata keys
            elif "usage_metadata" in metadata:
                usage = metadata["usage_metadata"]
                total_tokens += usage.get("total_tokens", 0)
                llm_calls += 1

    log.info(
        "discuss_phase_completed",
        message_count=len(messages),
        llm_calls=llm_calls,
        total_tokens=total_tokens,
    )

    return messages, llm_calls, total_tokens
