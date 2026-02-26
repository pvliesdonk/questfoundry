"""Summarize phase for condensing discussion into brief."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from questfoundry.agents.prompts import get_seed_section_summarize_prompts, get_summarize_prompt
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import build_runnable_config, traceable
from questfoundry.providers.content import extract_text

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph
    from questfoundry.pipeline.size import SizeProfile

log = get_logger(__name__)

# Ordered sections for chunked summarization.
# Each section builds on prior sections' output.
SEED_SUMMARY_SECTIONS = ("entities", "dilemmas", "paths", "beats", "convergence")


@traceable(name="Summarize Phase", run_type="chain", tags=["phase:summarize"])
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str | None = None,
    stage_name: str = "dream",
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, int]:
    """Summarize a discussion into a compact brief.

    This is a single LLM call (not an agent) that takes the conversation
    history from the Discuss phase and produces a compact summary for
    the Serialize phase.

    Uses lower temperature (0.3) for more focused, consistent output.

    Args:
        model: Chat model to use (will be invoked with low temperature)
        messages: Conversation history from Discuss phase
        system_prompt: Optional custom system prompt. If not provided,
            uses the default summarize prompt.
        stage_name: Stage name for logging/tagging (default "dream")
        callbacks: LangChain callback handlers for logging LLM calls

    Returns:
        Tuple of (summary_text, tokens_used)
    """
    log.info("summarize_started", message_count=len(messages), stage=stage_name)

    # Use custom prompt if provided, otherwise use default
    if system_prompt is None:
        system_prompt = get_summarize_prompt()

    # Build the messages for the summarize call
    # We include the system prompt, then the conversation as context,
    # then ask for the summary
    summarize_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content="Here is the discussion to summarize:\n\n"
            + _format_messages_for_summary(messages)
        ),
    ]

    # Build tracing config for the LLM call
    config = build_runnable_config(
        run_name="Summarize LLM Call",
        tags=[stage_name, "summarize", "llm"],
        metadata={"stage": stage_name, "phase": "summarize", "message_count": len(messages)},
        callbacks=callbacks,
    )

    # Note: We use the model as configured rather than trying to override temperature
    # at runtime. The bind(temperature=X) approach is not compatible with all providers
    # (e.g., langchain-ollama doesn't support runtime temperature in chat()).
    # The model's default temperature (0.7) works fine for summarization.
    response = await model.ainvoke(summarize_messages, config=config)

    # Extract the summary text
    summary = extract_text(response.content)

    # Extract token usage
    tokens = _extract_tokens(response) if isinstance(response, AIMessage) else 0

    log.info("summarize_completed", summary_length=len(summary), tokens=tokens)

    return summary, tokens


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Format conversation messages for the summary prompt.

    Preserves tool call context by including tool invocations and their results.
    This ensures research insights from tools (like corpus searches) are available
    in the summarized output.

    Output format:
        - Human messages: "User: <content>"
        - AI messages: "Assistant: <content>"
        - Tool calls: "[Tool Call: <name>]\\n<json args>"
        - Tool results: "[Tool Result: <name>]\\n<content>"
        - System messages: "System: <content>"

    Args:
        messages: List of conversation messages

    Returns:
        Formatted string representation of the conversation
    """
    formatted_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = extract_text(msg.content)
            formatted_parts.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            # Include text content if present and non-empty
            if msg.content:
                content = extract_text(msg.content)
                # Skip whitespace-only content to avoid noisy "Assistant:  " lines
                if content.strip():
                    formatted_parts.append(f"Assistant: {content}")
            # Include tool calls if present (research decisions made by the model)
            # tool_calls is a standard AIMessage attribute, but check type for safety
            if msg.tool_calls and isinstance(msg.tool_calls, list):
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    args_str = json.dumps(tool_args, indent=2)
                    formatted_parts.append(f"[Tool Call: {tool_name}]\n{args_str}")
        elif isinstance(msg, ToolMessage):
            # Include tool results (research findings) - extract just the useful content
            # to avoid prompt-stuffing with full JSON boilerplate
            tool_name = msg.name or "unknown_tool"
            raw_content = extract_text(msg.content)
            useful_content = raw_content
            try:
                data = json.loads(raw_content)
                # Try to extract the useful content, not the JSON wrapper
                if extracted := (data.get("content") or data.get("data")):
                    if isinstance(extracted, str):
                        useful_content = extracted
                    else:
                        useful_content = json.dumps(extracted, indent=2)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, stick with the raw content
                pass
            formatted_parts.append(f"[Research: {tool_name}]\n{useful_content}")
        elif isinstance(msg, SystemMessage):
            content = extract_text(msg.content)
            formatted_parts.append(f"System: {content}")
        else:
            content = extract_text(msg.content)
            formatted_parts.append(f"Message: {content}")

    return "\n\n".join(formatted_parts)


def _extract_tokens(response: AIMessage) -> int:
    """Extract total token count from an AIMessage response.

    Checks usage_metadata (Ollama) first, then response_metadata (OpenAI).

    Args:
        response: AIMessage from model.ainvoke().

    Returns:
        Total token count, or 0 if not available.
    """
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return response.usage_metadata.get("total_tokens") or 0
    if hasattr(response, "response_metadata") and response.response_metadata:
        metadata = response.response_metadata
        if "token_usage" in metadata:
            return metadata["token_usage"].get("total_tokens") or 0
    return 0


def _format_dilemma_answers_from_graph(graph: Graph) -> str:
    """Format valid answer IDs per dilemma from the graph.

    Builds a concise listing of each dilemma and its valid answer IDs,
    for injection into the dilemmas summarize section prompt.

    Args:
        graph: Graph containing brainstorm dilemma and answer nodes.

    Returns:
        Formatted string listing answer IDs per dilemma.
    """
    from questfoundry.graph.context import SCOPE_DILEMMA, normalize_scoped_id

    dilemmas = graph.get_nodes_by_type("dilemma")
    if not dilemmas:
        return "(No dilemmas)"

    # Pre-build answer edges map
    answer_edges_by_dilemma: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get_edges(edge_type="has_answer"):
        from_id = edge.get("from")
        if from_id:
            answer_edges_by_dilemma.setdefault(from_id, []).append(edge)

    lines: list[str] = []
    for did, ddata in sorted(dilemmas.items()):
        raw_id = ddata.get("raw_id")
        if not raw_id:
            continue

        answers: list[str] = []
        for edge in answer_edges_by_dilemma.get(did, []):
            answer_node = graph.get_node(edge.get("to", ""))
            if answer_node:
                ans_id = answer_node.get("raw_id")
                if ans_id:
                    default = " (default)" if answer_node.get("is_canonical") else ""
                    answers.append(f"`{ans_id}`{default}")

        if answers:
            answers.sort()
            scoped = normalize_scoped_id(raw_id, SCOPE_DILEMMA)
            lines.append(f"- `{scoped}` -> valid answers: [{', '.join(answers)}]")

    return "\n".join(lines) if lines else "(No dilemma answers)"


@traceable(name="Summarize Seed Chunked", run_type="chain", tags=["phase:summarize", "chunked"])
async def summarize_seed_chunked(
    model: BaseChatModel,
    messages: list[BaseMessage],
    graph: Graph,
    *,
    size_profile: SizeProfile | None = None,
    output_language_instruction: str = "",
    stage_name: str = "seed",
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[dict[str, str], int]:
    """Summarize SEED discussion into per-section briefs.

    Instead of one monolithic summarize call producing ~31K chars, this
    makes 5 sequential calls — one per section (entities, dilemmas, paths,
    beats, convergence). Each call receives the discussion history plus
    prior sections' output as context. The result is a dict of ~2-8K
    briefs that downstream serialize calls can use individually.

    Args:
        model: Chat model for summarization.
        messages: Conversation history from the Discuss phase.
        graph: Graph containing brainstorm data (for manifests and answer IDs).
        size_profile: Size profile for parameterizing count guidance.
        output_language_instruction: Language instruction for non-English output.
        stage_name: Stage name for logging/tagging.
        callbacks: LangChain callback handlers for logging.

    Returns:
        Tuple of (section_briefs, total_tokens) where section_briefs maps
        section name to its summarized brief text.
    """
    from questfoundry.graph.context import format_summarize_manifest, get_expected_counts

    log.info(
        "summarize_seed_chunked_started",
        message_count=len(messages),
        sections=len(SEED_SUMMARY_SECTIONS),
    )

    # Gather manifest data from graph
    counts = get_expected_counts(graph)
    manifests = format_summarize_manifest(graph)
    dilemma_answers = _format_dilemma_answers_from_graph(graph)

    # Build per-section system prompts
    section_prompts = get_seed_section_summarize_prompts(
        entity_count=counts["entities"],
        dilemma_count=counts["dilemmas"],
        entity_manifest=manifests["entity_manifest"],
        dilemma_manifest=manifests["dilemma_manifest"],
        dilemma_answers=dilemma_answers,
        size_profile=size_profile,
        output_language_instruction=output_language_instruction,
    )

    # Format discussion once (reused across all section calls)
    formatted_discussion = _format_messages_for_summary(messages)

    section_briefs: dict[str, str] = {}
    total_tokens = 0

    for section in SEED_SUMMARY_SECTIONS:
        system_prompt = section_prompts[section]

        # Build user message: discussion + prior sections' output
        user_parts = [
            "Here is the discussion to summarize:\n\n" + formatted_discussion,
        ]

        # Inject prior sections' output as context (dict preserves insertion order)
        if section_briefs:
            prior_context_parts = []
            for prev_section, prev_brief in section_briefs.items():
                prior_context_parts.append(
                    f"### {prev_section.title()} (already decided)\n{prev_brief}"
                )
            user_parts.append("\n\n---\n\n## Prior Decisions\n" + "\n\n".join(prior_context_parts))

        user_parts.append(f"\n\nNow summarize ONLY the **{section}** section from the discussion.")

        call_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(user_parts)),
        ]

        config = build_runnable_config(
            run_name=f"Summarize Seed Section: {section}",
            tags=[stage_name, "summarize", "chunked", section],
            metadata={
                "stage": stage_name,
                "phase": "summarize",
                "section": section,
                "message_count": len(messages),
            },
            callbacks=callbacks,
        )

        log.debug("summarize_seed_section_start", section=section)
        response = await model.ainvoke(call_messages, config=config)

        brief_text = extract_text(response.content)
        tokens = _extract_tokens(response) if isinstance(response, AIMessage) else 0

        section_briefs[section] = brief_text
        total_tokens += tokens

        log.debug(
            "summarize_seed_section_done",
            section=section,
            brief_length=len(brief_text),
            tokens=tokens,
        )

    log.info(
        "summarize_seed_chunked_completed",
        total_brief_length=sum(len(b) for b in section_briefs.values()),
        total_tokens=total_tokens,
        sections=list(section_briefs.keys()),
    )

    return section_briefs, total_tokens
