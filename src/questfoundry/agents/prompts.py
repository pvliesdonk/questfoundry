"""Prompt templates for agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from questfoundry.prompts.loader import PromptLoader


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


def get_discuss_prompt(
    research_tools_available: bool = True,
) -> str:
    """Build the Discuss phase prompt as a system message string.

    Loads the prompt template from prompts/templates/discuss.yaml and
    renders it with the provided context.

    The user_prompt is NOT included in the system message - it's passed
    separately as the initial HumanMessage to avoid duplication.

    Args:
        research_tools_available: Whether research tools are available

    Returns:
        System prompt string for the Discuss agent
    """
    loader = PromptLoader(_get_prompts_path())
    template = loader.load("discuss")

    # Build the research tools section if tools are available
    research_section = ""
    if research_tools_available:
        # Load the research tools section from the template
        raw_data = _load_raw_template("discuss")
        research_section = raw_data.get("research_tools_section", "")

    # Render the system template with Jinja2
    jinja_template = Template(template.system)
    return str(jinja_template.render(research_tools_section=research_section))


def get_summarize_prompt() -> str:
    """Build the Summarize phase prompt as a system message string.

    Loads the prompt template from prompts/templates/summarize.yaml.
    The summarize phase takes a conversation history and produces a
    compact brief for the serialize phase.

    Returns:
        System prompt string for the Summarize call
    """
    loader = PromptLoader(_get_prompts_path())
    template = loader.load("summarize")
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
