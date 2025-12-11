"""Prompt building with clean domain/runtime separation.

This module implements the domain-first prompt architecture:
- Domain layer: Role identity, mandate, constraints, anti-patterns (from MyST)
- Runtime layer: LLM-specific enforcement nudges, tool examples, stop conditions

The domain content comes from:
1. Compiled RoleIR definitions (generated from domain/roles/*.md)
2. The role's {role-prompt} template (Jinja2)

The runtime nudges are LLM engineering concerns:
- Tool call format examples
- Artifact handoff patterns
- Stop condition reminders
- Anti-pattern enforcement

Usage
-----
    from questfoundry.runtime.prompts import build_sr_prompt

    prompt = build_sr_prompt(roles)  # Combines domain + runtime nudges
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import BaseLoader, Environment

if TYPE_CHECKING:
    from questfoundry.compiler.models import RoleIR


# =============================================================================
# Runtime Nudges (LLM-specific enforcement, NOT domain knowledge)
# =============================================================================

SR_RUNTIME_NUDGES = """
## Runtime Guidelines

These are operational guidelines for the LLM runtime, not role definition.

### Output Format

Your responses should primarily consist of:
1. **Tool calls** - delegate_to(), write_artifact(), read_artifact(), terminate()
2. **Brief status updates** - 1-2 sentences explaining what you're doing

DO NOT write paragraphs of narrative content. If you find yourself writing
story prose, character descriptions, or scene details, STOP and delegate
to the appropriate role instead.

### Tool Call Examples

**Starting a workflow:**
```
consult_playbook("story spark workflow")
```

**Delegating creative work:**
```
delegate_to(
    role="plotwright",
    task="Design a 3-act mystery structure with red herrings",
    artifacts=[]
)
```

**Passing artifacts between roles:**
```
delegate_to(
    role="gatekeeper",
    task="Validate the story structure",
    artifacts=["act_1", "act_2", "act_3"]  # IDs from previous delegation
)
```

**Completing workflow:**
```
terminate(reason="All acts validated and promoted to canon")
```

### Artifact Handoff Pattern

When a role creates artifacts, the DelegationResult includes artifact IDs.
You MUST pass these to subsequent roles that need them:

1. Plotwright creates → artifacts: ["act_1", "act_2"]
2. You delegate to Gatekeeper with artifacts=["act_1", "act_2"]
3. Gatekeeper validates → artifacts: ["gatecheck_report_1"]
4. You delegate to Lorekeeper for canon promotion

### Stop Conditions

You should call `terminate()` when:
- All requested work is complete
- All artifacts are validated and promoted to canon
- You've received confirmation from Lorekeeper of canon promotion

You should NOT terminate when:
- Content is still in hot_store awaiting validation
- Gatekeeper has not reviewed the artifacts
- Creative work is still in progress

### Critical Anti-Pattern Reminder

**NEVER generate story content yourself.** If your response contains more than
2-3 sentences that aren't a tool call or status update, you're doing it wrong.

Example of WRONG behavior:
```
The detective entered the dimly lit room, her footsteps echoing on the
marble floor. She noticed the grandfather clock had stopped at 3:47...
```

Example of CORRECT behavior:
```
I'll delegate the scene writing to Scene Smith.
[delegate_to(role="scene_smith", task="Write the detective's entrance scene...")]
```
"""


# =============================================================================
# Domain Template Rendering
# =============================================================================


def _render_role_prompt_template(role: RoleIR) -> str:
    """Render the role's Jinja2 prompt template with role data.

    The domain file has a {role-prompt} directive with a Jinja2 template.
    This renders that template with the role's metadata.
    """
    if not role.prompt_template:
        # Fallback if no template defined
        return f"""You are the **{role.archetype}**.

Your mandate: **{role.mandate}**

## Constraints

{chr(10).join(f"- {c}" for c in role.constraints)}
"""

    # Create Jinja2 environment and render
    env = Environment(loader=BaseLoader())
    try:
        template = env.from_string(role.prompt_template)

        # Build context for template
        context = {
            "role": {
                "id": role.id,
                "abbr": role.abbr,
                "archetype": role.archetype,
                "agency": role.agency,
                "mandate": role.mandate,
                "tools": [{"name": t.name, "description": t.description} for t in role.tools],
                "constraints": role.constraints,
            }
        }

        return template.render(**context)
    except Exception as e:
        # If template rendering fails, use fallback
        return f"You are the {role.archetype}. Mandate: {role.mandate}\n\nError rendering template: {e}"


def _build_role_menu(roles: dict[str, RoleIR]) -> str:
    """Build the role menu table for SR's reference."""
    lines = [
        "| Code | Role ID | Archetype | Mandate |",
        "|------|---------|-----------|---------|",
    ]

    for role_id, role in sorted(roles.items()):
        if role_id == "showrunner":
            continue  # Don't list self
        lines.append(f"| {role.abbr} | **{role_id}** | {role.archetype} | {role.mandate} |")

    return "\n".join(lines)


def _build_sr_tools_section(roles: dict[str, RoleIR]) -> str:
    """Build the tools section dynamically from SR role definition."""
    sr_role = roles.get("showrunner")
    if not sr_role:
        return ""

    lines = ["## Your Tools", ""]

    # Group tools by category
    orchestration_tools = ["delegate_to", "terminate"]
    state_tools = ["read_artifact", "write_artifact", "list_hot_store_keys", "list_cold_store_keys"]
    knowledge_tools = ["consult_playbook", "consult_role_charter", "consult_schema"]

    lines.append("### Orchestration")
    for tool in sr_role.tools:
        if tool.name in orchestration_tools:
            lines.append(f"- **{tool.name}**: {tool.description}")

    lines.append("")
    lines.append("### State Management")
    for tool in sr_role.tools:
        if tool.name in state_tools:
            lines.append(f"- **{tool.name}**: {tool.description}")

    lines.append("")
    lines.append("### Knowledge (Consult Before Acting)")
    for tool in sr_role.tools:
        if tool.name in knowledge_tools:
            lines.append(f"- **{tool.name}**: {tool.description}")

    return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================


def build_sr_prompt(roles: dict[str, RoleIR]) -> str:
    """Build SR's complete system prompt with domain + runtime layers.

    This composes:
    1. Domain layer: Role identity from compiled RoleIR
    2. Role menu: Available specialist roles
    3. Runtime nudges: LLM-specific enforcement

    Parameters
    ----------
    roles : dict[str, RoleIR]
        All role definitions indexed by role_id.

    Returns
    -------
    str
        Complete system prompt for the Showrunner.
    """
    sr_role = roles.get("showrunner")
    if not sr_role:
        # Fallback if SR not defined
        return "You are the Showrunner. Coordinate work by delegating to specialists."

    # 1. Domain layer: Render role's prompt template
    domain_prompt = _render_role_prompt_template(sr_role)

    # 2. Add role menu
    role_menu = _build_role_menu(roles)

    # 3. Add runtime nudges
    runtime_nudges = SR_RUNTIME_NUDGES

    # Compose final prompt
    return f"""{domain_prompt}

## Available Specialist Roles

{role_menu}

{runtime_nudges}
"""


def build_role_prompt(role: RoleIR) -> str:
    """Build a specialist role's system prompt.

    For non-SR roles, this just renders the domain template without
    the orchestration-specific runtime nudges.

    Parameters
    ----------
    role : RoleIR
        Role definition.

    Returns
    -------
    str
        System prompt for the role.
    """
    return _render_role_prompt_template(role)
