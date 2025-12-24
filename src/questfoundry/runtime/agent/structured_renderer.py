"""
Renderers for structured knowledge content.

Converts semantic types (rule, contract, criterion, etc.) to markdown
for agent consumption. The structured JSON is the source of truth;
markdown is the presentation format.
"""

from __future__ import annotations

from typing import Any


def render_rule(rule: dict[str, Any]) -> str:
    """Render a rule to markdown."""
    lines = []

    statement = rule.get("statement", "")
    if not statement:
        return ""  # Skip rules without statements

    severity = rule.get("severity", "error")
    enforcement = rule.get("enforcement", "llm")

    # Header with severity indicator
    severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡"}.get(severity, "")
    lines.append(f"**{severity_icon} {statement}**")

    if enforcement == "runtime":
        lines.append("*(Enforced by runtime)*")

    if reasoning := rule.get("reasoning"):
        lines.append(f"\n{reasoning}")

    if examples := rule.get("examples"):
        lines.append("\n**Do:**")
        for ex in examples:
            lines.append(f"- {ex}")

    if counter_examples := rule.get("counter_examples"):
        lines.append("\n**Don't:**")
        for ex in counter_examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def render_contract(contract: dict[str, Any]) -> str:
    """Render a contract to markdown."""
    lines = []

    name = contract.get("name", "Contract")
    parties = contract.get("parties") or []
    obligations = contract.get("obligations", [])

    lines.append(f"### {name}")
    if parties:
        lines.append(f"**Parties:** {', '.join(parties)}")
    else:
        lines.append("**Parties:** _None defined_")

    if obligations:
        lines.append("\n**Obligations:**")
        for obl in obligations:
            party = obl.get("party", "?")
            must = obl.get("must", "")
            when = obl.get("when")
            if when:
                lines.append(f"- **{party}** must: {must} *(when: {when})*")
            else:
                lines.append(f"- **{party}** must: {must}")

    if reasoning := contract.get("reasoning"):
        lines.append(f"\n*{reasoning}*")

    if examples := contract.get("examples"):
        lines.append("\n**Examples:**")
        for ex in examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def render_criterion(criterion: dict[str, Any]) -> str:
    """Render a criterion to markdown."""
    lines = []

    name = criterion.get("name", "Criterion")
    pass_condition = criterion.get("pass_condition", "")

    lines.append(f"### {name}")
    lines.append(f"**Pass when:** {pass_condition}")

    if fail_indicators := criterion.get("fail_indicators"):
        lines.append("\n**Fail indicators:**")
        for ind in fail_indicators:
            lines.append(f"- {ind}")

    if reasoning := criterion.get("reasoning"):
        lines.append(f"\n*{reasoning}*")

    if examples := criterion.get("examples"):
        lines.append("\n**Good examples:**")
        for ex in examples:
            lines.append(f"- {ex}")

    if counter_examples := criterion.get("counter_examples"):
        lines.append("\n**Bad examples:**")
        for ex in counter_examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def render_heuristic(heuristic: dict[str, Any]) -> str:
    """Render a heuristic to markdown."""
    lines = []

    context = heuristic.get("context")
    guidance = heuristic.get("guidance", "")

    if context:
        lines.append(f"**When:** {context}")
    lines.append(f"**Guidance:** {guidance}")

    if reasoning := heuristic.get("reasoning"):
        lines.append(f"\n*{reasoning}*")

    if examples := heuristic.get("examples"):
        lines.append("\n**Examples:**")
        for ex in examples:
            lines.append(f"- {ex}")

    if counter_examples := heuristic.get("counter_examples"):
        lines.append("\n**Avoid:**")
        for ex in counter_examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def render_definition(definition: dict[str, Any]) -> str:
    """Render a definition to markdown."""
    lines = []

    term = definition.get("term", "")
    meaning = definition.get("meaning", "")

    lines.append(f"**{term}**: {meaning}")

    if features := definition.get("distinguishing_features"):
        for feat in features:
            lines.append(f"- {feat}")

    if examples := definition.get("examples"):
        lines.append("\n**Examples:**")
        for ex in examples:
            lines.append(f"- {ex}")

    if counter_examples := definition.get("counter_examples"):
        lines.append("\n**Not this:**")
        for ex in counter_examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def render_procedure(procedure: dict[str, Any]) -> str:
    """Render a procedure to markdown."""
    lines = []

    goal = procedure.get("goal", "Procedure")
    steps = procedure.get("steps", [])

    lines.append(f"### {goal}")

    if preconditions := procedure.get("preconditions"):
        lines.append("\n**Preconditions:**")
        for pre in preconditions:
            lines.append(f"- {pre}")

    if steps:
        lines.append("\n**Steps:**")
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")

    if postconditions := procedure.get("postconditions"):
        lines.append("\n**Postconditions:**")
        for post in postconditions:
            lines.append(f"- {post}")

    if examples := procedure.get("examples"):
        lines.append("\n**Examples:**")
        for ex in examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def render_warning(warning: dict[str, Any]) -> str:
    """Render a warning to markdown."""
    lines = []

    failure_mode = warning.get("failure_mode", "")
    consequence = warning.get("consequence")
    detection = warning.get("detection")
    prevention = warning.get("prevention")

    lines.append(f"⚠️ **{failure_mode}**")

    if consequence:
        lines.append(f"\n*Consequence:* {consequence}")

    if detection:
        lines.append(f"\n*Detection:* {detection}")

    if prevention:
        lines.append(f"\n*Prevention:* {prevention}")

    if examples := warning.get("examples"):
        lines.append("\n**Do instead:**")
        for ex in examples:
            lines.append(f"- {ex}")

    if counter_examples := warning.get("counter_examples"):
        lines.append("\n**Avoid:**")
        for ex in counter_examples:
            lines.append(f"- {ex}")

    return "\n".join(lines)


def _render_section(
    items: list[dict[str, Any]] | None,
    header: str,
    renderer: Any,
) -> list[str]:
    """Render a section of items, filtering out empty renders."""
    if not items:
        return []

    rendered = [renderer(item) for item in items]
    # Filter out empty renders
    rendered = [r for r in rendered if r]

    if not rendered:
        return []

    result = [f"## {header}\n"]
    for r in rendered:
        result.append(r)
        result.append("")
    return result


def render_structured_entry(data: dict[str, Any]) -> str:
    """Render a complete structured entry to markdown.

    Takes the 'data' field from a structured knowledge content
    and renders all semantic types to readable markdown.

    Empty items (e.g., rules without statements) are filtered out.
    """
    sections: list[str] = []

    # Render each semantic type if present, filtering empty items
    sections.extend(_render_section(data.get("rules"), "Rules", render_rule))
    sections.extend(_render_section(data.get("contracts"), "Contracts", render_contract))
    sections.extend(_render_section(data.get("criteria"), "Criteria", render_criterion))
    sections.extend(_render_section(data.get("definitions"), "Definitions", render_definition))
    sections.extend(_render_section(data.get("procedures"), "Procedures", render_procedure))
    sections.extend(_render_section(data.get("heuristics"), "Guidance", render_heuristic))
    sections.extend(_render_section(data.get("warnings"), "Warnings", render_warning))

    if not sections:
        return "(No structured content available.)"

    return "\n".join(sections).strip()
