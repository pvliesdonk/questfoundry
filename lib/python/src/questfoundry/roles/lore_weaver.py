"""Lore Weaver role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class LoreWeaver(Role):
    """
    Lore Weaver: Transform accepted hooks into coherent canon.

    The Lore Weaver resolves causes and constraints by turning accepted hooks
    into canonical truths (backstories, timelines, metaphysics, causal links).
    They maintain spoiler hygiene by keeping sensitive lore in Hot and providing
    player-safe summaries to other roles.

    Key responsibilities:
    - Answer canonical questions from accepted hooks
    - Establish timelines, causal chains, and constraints
    - Resolve contradictions with existing canon
    - Provide player-safe summaries for Codex Curator
    - Support Plotwright and Scene Smith with lore notes
    - Maintain knowledge ledger (who knows what, when)
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/lore_weaver.md"""
        return "lore_weaver"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Lore Weaver"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a lore weaver task.

        Supported tasks:
        - 'expand_canon': Turn hooks into canonical lore entries
        - 'resolve_contradiction': Adjudicate conflicting canon
        - 'create_timeline': Establish temporal relationships
        - 'generate_player_summary': Create player-safe lore summary
        - 'check_canon_consistency': Validate against existing canon

        Args:
            context: Execution context

        Returns:
            Result with canon entries or validation results
        """
        task = context.task.lower()
        logger.info("LoreWeaver executing task: %s", task)
        logger.trace("Number of artifacts provided: %d", len(context.artifacts))

        if task == "expand_canon":
            logger.debug("Expanding canon from hooks")
            return self._expand_canon(context)
        elif task == "resolve_contradiction":
            logger.debug("Resolving canon contradiction")
            return self._resolve_contradiction(context)
        elif task == "create_timeline":
            logger.debug("Creating temporal timeline")
            return self._create_timeline(context)
        elif task == "generate_player_summary":
            logger.debug("Generating player-safe summary")
            return self._generate_player_summary(context)
        elif task == "check_canon_consistency":
            logger.debug("Checking canon consistency")
            return self._check_canon_consistency(context)
        else:
            logger.warning("Unknown LoreWeaver task: %s", task)
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _expand_canon(self, context: RoleContext) -> RoleResult:
        """Turn accepted hooks into canonical lore entries."""
        system_prompt = self.build_system_prompt(context)

        hooks = context.additional_context.get("hooks", [])
        cluster_name = context.additional_context.get("cluster_name", "")

        user_prompt = f"""# Task: Expand Canon from Hooks

{self.format_artifacts(context.artifacts)}

## Cluster: {cluster_name}

## Accepted Hooks
{self._format_hooks(hooks)}

Transform these hooks into coherent canon entries. For each hook or cluster:

1. **Canon Answer**: Precise, spoiler-level answer to the question
2. **Timeline Anchors**: When did this happen? What's the sequence?
3. **Causal Links**: What caused this? What did it cause?
4. **Entities Affected**: Who/what is involved or changed?
5. **Constraints**: What cannot happen because of this truth?
6. **Knowledge Ledger**: Who knows what, and when did they learn it?

Provide responses in JSON format:
```json
{{
  "entries": [
    {{
      "title": "Canon title",
      "answer": "Detailed canonical answer",
      "timeline": ["Anchor 1", "Anchor 2"],
      "causal_links": ["Cause 1", "Effect 1"],
      "entities": ["Entity 1", "Entity 2"],
      "constraints": ["Cannot X", "Must Y"],
      "sensitivity": "spoiler-heavy|player-safe-summary-possible",
      "player_safe_summary": "Neutral, non-revealing summary",
      "downstream_impacts": {{
        "plotwright": "Gate/topology implications",
        "scene_smith": "Scene callback notes",
        "style_lead": "Motif opportunities"
      }}
    }}
  ]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)

            # Parse JSON response
            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "canon_pack",
                    "entries": data.get("entries", []),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error expanding canon: {e}",
            )

    def _resolve_contradiction(self, context: RoleContext) -> RoleResult:
        """Adjudicate conflicting canon statements."""
        system_prompt = self.build_system_prompt(context)

        contradiction = context.additional_context.get("contradiction", {})

        user_prompt = f"""# Task: Resolve Contradiction

{self.format_artifacts(context.artifacts)}

## Conflicting Statements
Statement A: {contradiction.get("statement_a", "")}
Source A: {contradiction.get("source_a", "")}

Statement B: {contradiction.get("statement_b", "")}
Source B: {contradiction.get("source_b", "")}

Resolve this contradiction by:
1. **Analysis**: What's the core conflict?
2. **Resolution**: Which statement takes precedence, or how can both be true?
3. **Rationale**: Why this resolution?
4. **Downstream Changes**: What needs updating?

Respond in JSON format with resolution decision.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "contradiction_resolution",
                    "resolution": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error resolving contradiction: {e}",
            )

    def _create_timeline(self, context: RoleContext) -> RoleResult:
        """Establish temporal relationships for events."""
        system_prompt = self.build_system_prompt(context)

        events = context.additional_context.get("events", [])

        user_prompt = f"""# Task: Create Timeline

{self.format_artifacts(context.artifacts)}

## Events to Sequence
{self._format_list(events)}

Create a timeline that establishes:
1. **Sequence**: What happened in what order?
2. **Anchors**: Fixed points (dates, durations, relative times)
3. **Dependencies**: What must happen before what?
4. **Gaps**: Known unknowns or mysteries

Respond in JSON format with timeline structure.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "timeline", "timeline": data},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating timeline: {e}",
            )

    def _generate_player_summary(self, context: RoleContext) -> RoleResult:
        """Create player-safe summary from spoiler-heavy canon."""
        system_prompt = self.build_system_prompt(context)

        canon_entry = context.additional_context.get("canon_entry", {})

        user_prompt = f"""# Task: Generate Player-Safe Summary

{self.format_artifacts(context.artifacts)}

## Canon Entry (Spoiler-Heavy)
Title: {canon_entry.get("title", "")}
Answer: {canon_entry.get("answer", "")}

Create a player-safe summary that:
1. Provides useful context without revealing twists
2. Uses neutral phrasing (no tease lines)
3. Focuses on surface-level facts
4. Avoids causal chains that hint at solutions

Keep it 2-4 sentences, bland and informative.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=500)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "player_safe_summary"},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error generating player summary: {e}",
            )

    def _check_canon_consistency(self, context: RoleContext) -> RoleResult:
        """Validate canon against existing truths."""
        system_prompt = self.build_system_prompt(context)

        new_canon = context.additional_context.get("new_canon", {})

        user_prompt = f"""# Task: Check Canon Consistency

{self.format_artifacts(context.artifacts)}

## New Canon to Validate
{self._format_dict(new_canon)}

Check for:
1. **Contradictions**: Conflicts with existing canon
2. **Timeline Issues**: Impossible sequences or overlaps
3. **Entity Conflicts**: Character/location inconsistencies
4. **Constraint Violations**: Breaking established rules

Respond in JSON format:
```json
{{
  "is_consistent": true|false,
  "issues": [
    {{
      "type": "contradiction|timeline|entity|constraint",
      "description": "Issue description",
      "severity": "blocker|warning|info"
    }}
  ],
  "recommendations": ["Action 1", "Action 2"]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "consistency_check",
                    "is_consistent": data.get("is_consistent", False),
                    "issues": data.get("issues", []),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error checking consistency: {e}",
            )

    def _format_hooks(self, hooks: list[dict[str, Any]]) -> str:
        """Format hooks list for prompt."""
        if not hooks:
            return "(no hooks provided)"

        formatted = []
        for i, hook in enumerate(hooks, 1):
            formatted.append(
                f"{i}. {hook.get('title', 'Untitled')}: "
                f"{hook.get('summary', 'No summary')}"
            )
        return "\n".join(formatted)

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format dictionary as bullet list."""
        if not d:
            return "(empty)"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())

    def _format_list(self, items: list[str]) -> str:
        """Format list as bullet list."""
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)
