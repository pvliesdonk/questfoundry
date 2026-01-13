"""Prompt templates for agents.

Uses LangChain's ChatPromptTemplate for variable injection, with templates
stored externally in YAML files under prompts/templates/.
"""

from __future__ import annotations

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
) -> str:
    """Build the Discuss phase prompt as a system message string.

    Loads the prompt template from prompts/templates/discuss.yaml and
    renders it with the provided context using ChatPromptTemplate.

    The user_prompt is NOT included in the system message - it's passed
    separately as the initial HumanMessage to avoid duplication.

    Args:
        research_tools_available: Whether research tools are available

    Returns:
        System prompt string for the Discuss agent
    """
    # Load raw data once to avoid double file read
    raw_data = _load_raw_template("discuss")

    # Build the research tools section if tools are available
    research_section = ""
    if research_tools_available:
        research_section = raw_data.get("research_tools_section", "")

    # Render the system template with ChatPromptTemplate
    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)
    return prompt.format(research_tools_section=research_section)


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


def get_brainstorm_discuss_prompt(
    vision_context: str,
    research_tools_available: bool = True,
) -> str:
    """Build the BRAINSTORM discuss prompt with vision context.

    Uses _load_raw_template() instead of _get_loader() because we need access
    to the 'research_tools_section' field which isn't in the PromptTemplate
    dataclass. Summarize prompts use _get_loader() since they only need system.

    Args:
        vision_context: Formatted vision from DREAM stage.
        research_tools_available: Whether research tools are available.

    Returns:
        System prompt string for the BRAINSTORM discuss agent.
    """
    raw_data = _load_raw_template("discuss_brainstorm")

    # Build research tools section
    research_section = ""
    if research_tools_available:
        research_section = raw_data.get("research_tools_section", "")

    # Render the system template
    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)
    return prompt.format(
        vision_context=vision_context,
        research_tools_section=research_section,
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
) -> str:
    """Build the SEED discuss prompt with brainstorm context.

    Uses _load_raw_template() to access the 'research_tools_section' field.

    Args:
        brainstorm_context: Formatted brainstorm output from BRAINSTORM stage.
        research_tools_available: Whether research tools are available.

    Returns:
        System prompt string for the SEED discuss agent.
    """
    raw_data = _load_raw_template("discuss_seed")

    # Build research tools section
    research_section = ""
    if research_tools_available:
        research_section = raw_data.get("research_tools_section", "")

    # Render the system template
    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)
    return prompt.format(
        brainstorm_context=brainstorm_context,
        research_tools_section=research_section,
    )


def get_seed_summarize_prompt() -> str:
    """Build the SEED summarize prompt.

    Returns:
        System prompt string for the SEED summarize call.
    """
    loader = _get_loader()
    template = loader.load("summarize_seed")
    return template.system
