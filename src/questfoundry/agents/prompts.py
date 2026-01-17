"""Prompt templates for agents.

Uses LangChain's ChatPromptTemplate for variable injection, with templates
stored externally in YAML files under prompts/templates/.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from questfoundry.prompts.loader import PromptLoader

# Module-level loader for caching efficiency
_prompt_loader: PromptLoader | None = None


def _get_prompts_path() -> Path:
    """Get the prompts directory path.

    Returns prompts from package first, then falls back to project root.
    """
    # Package prompts directory (installed)
    pkg_path = Path(__file__).parent.parent.parent.parent / "prompts"
    if pkg_path.exists():
        return pkg_path

    # Project root fallback (development)
    return Path.cwd() / "prompts"


def _get_loader() -> PromptLoader:
    """Get or create the module-level PromptLoader.

    Uses lazy initialization to allow path detection at first use.
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(_get_prompts_path())
    return _prompt_loader


def get_discuss_prompt(
    research_tools_available: bool = True,
    interactive: bool = True,
) -> str:
    """Build the Discuss phase prompt as a system message string.

    Loads the prompt template from prompts/templates/discuss.yaml and
    renders it with the provided context using ChatPromptTemplate.

    The user_prompt is NOT included in the system message - it's passed
    separately as the initial HumanMessage to avoid duplication.

    Args:
        research_tools_available: Whether research tools are available.
        interactive: Whether running in interactive mode. When False,
            includes instructions for autonomous decision-making.

    Returns:
        System prompt string for the Discuss agent
    """
    return _render_discuss_template(
        "discuss",
        research_tools_available=research_tools_available,
        interactive=interactive,
    )


def get_summarize_prompt() -> str:
    """Build the Summarize phase prompt as a system message string.

    Loads the prompt template from prompts/templates/summarize.yaml.
    The summarize phase takes a conversation history and produces a
    compact brief for the serialize phase.

    Returns:
        System prompt string for the Summarize call
    """
    loader = _get_loader()
    template = loader.load("summarize")
    return template.system


def get_serialize_prompt() -> str:
    """Build the Serialize phase prompt as a system message string.

    Loads the prompt template from prompts/templates/serialize.yaml.
    The serialize phase takes a brief and produces a structured artifact.

    Returns:
        System prompt string for the Serialize call
    """
    loader = _get_loader()
    template = loader.load("serialize")
    return template.system


def _load_raw_template(template_name: str) -> dict[str, Any]:
    """Load raw template data without parsing into PromptTemplate.

    This is needed to access fields not in the PromptTemplate dataclass.
    """
    from ruamel.yaml import YAML

    path = _get_prompts_path() / "templates" / f"{template_name}.yaml"
    yaml = YAML()
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.load(f))


def _render_discuss_template(
    template_name: str,
    research_tools_available: bool,
    interactive: bool,
    **kwargs: Any,
) -> str:
    """Render a discuss-style prompt template.

    Common helper for discuss prompts that share the same pattern:
    load template, build research/mode sections, format with kwargs.

    Args:
        template_name: Name of the template file (without .yaml).
        research_tools_available: Whether to include research tools section.
        interactive: Whether running interactively. When False, includes
            autonomous decision-making instructions.
        **kwargs: Additional format arguments for the template.

    Returns:
        Rendered system prompt string.
    """
    raw_data = _load_raw_template(template_name)

    research_section = (
        raw_data.get("research_tools_section", "") if research_tools_available else ""
    )
    mode_section = raw_data.get("non_interactive_section", "") if not interactive else ""

    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)

    return prompt.format(
        research_tools_section=research_section,
        mode_section=mode_section,
        **kwargs,
    )


def get_brainstorm_discuss_prompt(
    vision_context: str,
    research_tools_available: bool = True,
    interactive: bool = True,
) -> str:
    """Build the BRAINSTORM discuss prompt with vision context.

    Args:
        vision_context: Formatted vision from DREAM stage.
        research_tools_available: Whether research tools are available.
        interactive: Whether running in interactive mode. When False,
            includes instructions for autonomous decision-making.

    Returns:
        System prompt string for the BRAINSTORM discuss agent.
    """
    return _render_discuss_template(
        "discuss_brainstorm",
        research_tools_available=research_tools_available,
        interactive=interactive,
        vision_context=vision_context,
    )


def get_brainstorm_summarize_prompt() -> str:
    """Build the BRAINSTORM summarize prompt.

    Returns:
        System prompt string for the BRAINSTORM summarize call.
    """
    loader = _get_loader()
    template = loader.load("summarize_brainstorm")
    return template.system


def get_seed_discuss_prompt(
    brainstorm_context: str,
    research_tools_available: bool = True,
    interactive: bool = True,
) -> str:
    """Build the SEED discuss prompt with brainstorm context.

    Args:
        brainstorm_context: Formatted brainstorm output from BRAINSTORM stage.
        research_tools_available: Whether research tools are available.
        interactive: Whether running in interactive mode. When False,
            includes instructions for autonomous decision-making.

    Returns:
        System prompt string for the SEED discuss agent.
    """
    return _render_discuss_template(
        "discuss_seed",
        research_tools_available=research_tools_available,
        interactive=interactive,
        brainstorm_context=brainstorm_context,
    )


def get_seed_summarize_prompt(brainstorm_context: str = "") -> str:
    """Build the SEED summarize prompt.

    Args:
        brainstorm_context: YAML representation of brainstorm entities/tensions.
            Required for the summarizer to know what IDs to reference.

    Returns:
        System prompt string for the SEED summarize call.
    """
    raw_data = _load_raw_template("summarize_seed")

    # Render the system template with brainstorm context
    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)
    return prompt.format(brainstorm_context=brainstorm_context)


def get_brainstorm_serialize_prompt() -> str:
    """Build the BRAINSTORM serialize prompt.

    This prompt includes explicit instructions for mapping prose categories
    (Characters, Locations, Objects, Factions) to Entity objects with the
    correct type field.

    Returns:
        System prompt string for the BRAINSTORM serialize call.
    """
    loader = _get_loader()
    template = loader.load("serialize_brainstorm")
    return template.system


def get_seed_serialize_prompt() -> str:
    """Build the SEED serialize prompt.

    This prompt includes explicit instructions for mapping the complex
    SEED brief sections (Entity Decisions, Tension Decisions, Threads,
    Consequences, Initial Beats, Convergence Sketch) to the SeedOutput schema.

    Returns:
        System prompt string for the SEED serialize call.
    """
    loader = _get_loader()
    template = loader.load("serialize_seed")
    return template.system


def get_repair_seed_brief_prompt(
    valid_ids_context: str,
    error_list: str,
    brief: str,
) -> tuple[str, str]:
    """Build the repair SEED brief prompt.

    This prompt instructs the model to surgically fix invalid ID references
    in a brief without changing any other content.

    Args:
        valid_ids_context: Formatted list of valid IDs from BRAINSTORM.
        error_list: Formatted list of semantic validation errors with suggestions.
        brief: The original brief with invalid IDs to repair.

    Returns:
        Tuple of (system_prompt, user_prompt) for the repair call.
    """
    raw_data = _load_raw_template("repair_seed_brief")

    system_template = raw_data.get("system", "")
    user_template = raw_data.get("user", "")

    system_prompt = ChatPromptTemplate.from_template(system_template)
    user_prompt = ChatPromptTemplate.from_template(user_template)

    return (
        system_prompt.format(valid_ids_context=valid_ids_context, error_list=error_list),
        user_prompt.format(brief=brief),
    )


# --- Entity Coverage Validation ---


def _extract_entity_checklist(brainstorm_context: str) -> tuple[str, int]:
    """Extract entity IDs from brainstorm context for explicit checklist.

    Parses the "## Entities from BRAINSTORM" section and creates a
    numbered list for the LLM to process sequentially. Uses simple
    formatting that's harder to creatively reformat.

    Args:
        brainstorm_context: Formatted brainstorm context string containing
            entities in the format "- **entity_id** (type): concept".

    Returns:
        Tuple of (checklist_text, entity_count).
    """
    # Parse entity IDs from formatted context (lines like "- **entity_id** (type): concept")
    # Pattern allows letters, digits, underscores, hyphens in IDs
    pattern = r"\- \*\*([\w-]+)\*\* \((\w+)\):"
    matches = re.findall(pattern, brainstorm_context)

    if not matches:
        return "No entities found in brainstorm context.", 0

    # Group by type for clearer presentation
    by_type: dict[str, list[str]] = {}
    for entity_id, entity_type in matches:
        by_type.setdefault(entity_type, []).append(entity_id)

    # Use numbered list format - research shows this prevents skipping
    lines = []
    counter = 1
    for entity_type in sorted(by_type.keys()):
        ids = by_type[entity_type]
        lines.append(f"**{entity_type.title()}s** ({len(ids)}):")
        for eid in sorted(ids):
            lines.append(f"  {counter}. {eid}")
            counter += 1
        lines.append("")

    lines.append(f"Total: {len(matches)} entities. Process each one sequentially.")

    return "\n".join(lines), len(matches)


def _count_tensions(brainstorm_context: str) -> int:
    """Count tensions in brainstorm context.

    Args:
        brainstorm_context: Formatted brainstorm context string.

    Returns:
        Number of tensions found.
    """
    # Tensions are formatted as "- **tension_id**: question"
    # Pattern allows letters, digits, underscores, hyphens in IDs
    pattern = r"\- \*\*([\w-]+)\*\*:"

    # Look in the tensions section only, stopping at next section header
    tensions_section = ""
    start = brainstorm_context.find("## Tensions from BRAINSTORM")
    if start != -1:
        end = brainstorm_context.find("\n## ", start + 1)
        tensions_section = (
            brainstorm_context[start:end] if end != -1 else brainstorm_context[start:]
        )

    matches = re.findall(pattern, tensions_section)
    return len(matches)


def validate_entity_coverage(brief: str, expected_count: int) -> tuple[bool, int]:
    """Validate that brief contains decisions for all expected entities.

    External validation after summarize - LLMs cannot self-verify mid-generation.
    This catches incomplete coverage before serialize phase.

    Args:
        brief: The summarized brief from the LLM.
        expected_count: Number of entities expected from brainstorm context.

    Returns:
        Tuple of (is_complete, actual_count).
        - is_complete: True if actual >= expected
        - actual_count: Number of entity decisions found
    """
    # Count entity decisions in brief
    # Format: "- id: entity_id" or "id: entity_id" at start of line
    # The summarize prompt asks for "id: entity_id_here" format
    # Pattern allows letters, digits, underscores, hyphens in IDs
    pattern = r"^\s*-?\s*id:\s*([\w-]+)"
    matches = re.findall(pattern, brief, re.MULTILINE | re.IGNORECASE)

    actual_count = len(matches)
    is_complete = actual_count >= expected_count

    return is_complete, actual_count


def get_expected_entity_count(brainstorm_context: str) -> int:
    """Get expected entity count from brainstorm context.

    Args:
        brainstorm_context: Formatted brainstorm context string.

    Returns:
        Number of entities expected.
    """
    _, count = _extract_entity_checklist(brainstorm_context)
    return count
