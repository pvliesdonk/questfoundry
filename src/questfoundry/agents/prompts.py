"""Prompt templates for agents.

Uses LangChain's ChatPromptTemplate for variable injection, with templates
stored externally in YAML files under prompts/templates/.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate

from questfoundry.pipeline.size import size_template_vars
from questfoundry.prompts.loader import PromptLoader

if TYPE_CHECKING:
    from questfoundry.pipeline.size import SizeProfile

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
    size_presets_section = raw_data.get("size_presets_section", "")

    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)

    return prompt.format(
        research_tools_section=research_section,
        mode_section=mode_section,
        size_presets_section=size_presets_section,
        **kwargs,
    )


def get_brainstorm_discuss_prompt(
    vision_context: str,
    research_tools_available: bool = True,
    interactive: bool = True,
    size_profile: SizeProfile | None = None,
) -> str:
    """Build the BRAINSTORM discuss prompt with vision context.

    Args:
        vision_context: Formatted vision from DREAM stage.
        research_tools_available: Whether research tools are available.
        interactive: Whether running in interactive mode. When False,
            includes instructions for autonomous decision-making.
        size_profile: Size profile for parameterizing count guidance.

    Returns:
        System prompt string for the BRAINSTORM discuss agent.
    """
    return _render_discuss_template(
        "discuss_brainstorm",
        research_tools_available=research_tools_available,
        interactive=interactive,
        vision_context=vision_context,
        **size_template_vars(size_profile),
    )


def get_brainstorm_summarize_prompt(
    size_profile: SizeProfile | None = None,
) -> str:
    """Build the BRAINSTORM summarize prompt.

    Args:
        size_profile: Size profile for parameterizing count guidance.

    Returns:
        System prompt string for the BRAINSTORM summarize call.
    """
    loader = _get_loader()
    template = loader.load("summarize_brainstorm")
    prompt = ChatPromptTemplate.from_template(template.system)
    return prompt.format(**size_template_vars(size_profile))


def get_seed_discuss_prompt(
    brainstorm_context: str,
    research_tools_available: bool = True,
    interactive: bool = True,
    size_profile: SizeProfile | None = None,
) -> str:
    """Build the SEED discuss prompt with brainstorm context.

    Args:
        brainstorm_context: Formatted brainstorm output from BRAINSTORM stage.
        research_tools_available: Whether research tools are available.
        interactive: Whether running in interactive mode. When False,
            includes instructions for autonomous decision-making.
        size_profile: Size profile for parameterizing count guidance.

    Returns:
        System prompt string for the SEED discuss agent.
    """
    return _render_discuss_template(
        "discuss_seed",
        research_tools_available=research_tools_available,
        interactive=interactive,
        brainstorm_context=brainstorm_context,
        **size_template_vars(size_profile),
    )


def get_seed_summarize_prompt(
    brainstorm_context: str = "",
    entity_count: int = 0,
    dilemma_count: int = 0,
    entity_manifest: str = "",
    dilemma_manifest: str = "",
    size_profile: SizeProfile | None = None,
) -> str:
    """Build the SEED summarize prompt with manifest awareness.

    The manifest parameters enable the summarizer to know exactly which IDs
    it must include decisions for, enforcing completeness by construction.

    Args:
        brainstorm_context: YAML representation of brainstorm entities/dilemmas.
            Required for the summarizer to know what IDs to reference.
        entity_count: Total number of entities requiring decisions.
        dilemma_count: Total number of dilemmas requiring decisions.
        entity_manifest: Formatted list of entity IDs for manifest.
        dilemma_manifest: Formatted list of dilemma IDs for manifest.
        size_profile: Size profile for parameterizing count guidance.

    Returns:
        System prompt string for the SEED summarize call.
    """
    raw_data = _load_raw_template("summarize_seed")

    # Render the system template with brainstorm context and manifest info
    system_template = raw_data.get("system", "")
    prompt = ChatPromptTemplate.from_template(system_template)
    return prompt.format(
        brainstorm_context=brainstorm_context,
        entity_count=entity_count,
        dilemma_count=dilemma_count,
        entity_manifest=entity_manifest or "(No entities)",
        dilemma_manifest=dilemma_manifest or "(No dilemmas)",
        **size_template_vars(size_profile),
    )


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


def get_seed_serialize_prompt(
    size_profile: SizeProfile | None = None,
) -> str:
    """Build the SEED serialize prompt.

    This prompt includes explicit instructions for mapping the complex
    SEED brief sections (Entity Decisions, Dilemma Decisions, Paths,
    Consequences, Initial Beats, Convergence Sketch) to the SeedOutput schema.

    Args:
        size_profile: Size profile for parameterizing count guidance.

    Returns:
        System prompt string for the SEED serialize call.
    """
    loader = _get_loader()
    template = loader.load("serialize_seed")
    prompt = ChatPromptTemplate.from_template(template.system)
    return prompt.format(**size_template_vars(size_profile))
