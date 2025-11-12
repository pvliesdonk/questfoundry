"""Codex Curator role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class CodexCurator(Role):
    """
    Codex Curator: Create player-safe codex entries from canon.

    The Codex Curator transforms canon (often spoiler-heavy) into player-safe
    codex pages with clear cross-references. They improve comprehension without
    leaking twists or internal plumbing.

    Key responsibilities:
    - Author player-safe codex entries from lore summaries
    - Create cross-reference networks (See also links)
    - Maintain terminology consistency
    - Ensure accessibility (headings, links, alt text)
    - Address taxonomy/clarity hooks
    - Coordinate with Translator on localization
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/codex_curator.md"""
        return "codex_curator"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Codex Curator"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a codex curator task.

        Supported tasks:
        - 'create_entry': Create a codex entry from player-safe summary
        - 'create_crosslinks': Generate See also links between entries
        - 'check_coverage': Identify missing or incomplete entries
        - 'validate_accessibility': Ensure accessibility standards
        - 'create_glossary_slice': Extract terminology for translation
        - 'create_snapshot': Create archive snapshot (alias for create_entry)
        - 'validate_snapshot': Validate snapshot (alias for validate_accessibility)

        Args:
            context: Execution context

        Returns:
            Result with codex entries or validation results
        """
        task = context.task.lower()

        if task == "create_entry":
            return self._create_entry(context)
        elif task == "create_crosslinks":
            return self._create_crosslinks(context)
        elif task == "check_coverage":
            return self._check_coverage(context)
        elif task == "validate_accessibility":
            return self._validate_accessibility(context)
        elif task == "create_glossary_slice":
            return self._create_glossary_slice(context)
        # New tasks for loops
        elif task == "create_snapshot":
            return self._create_entry(context)
        elif task == "validate_snapshot":
            return self._validate_accessibility(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _create_entry(self, context: RoleContext) -> RoleResult:
        """Create a codex entry from player-safe summary."""
        system_prompt = self.build_system_prompt(context)

        topic = context.additional_context.get("topic", "")
        summary = context.additional_context.get("summary", "")
        related_terms = context.additional_context.get("related_terms", [])

        user_prompt = f"""# Task: Create Codex Entry

{self.format_artifacts(context.artifacts)}

## Topic
{topic}

## Player-Safe Summary (from Lore Weaver)
{summary}

## Related Terms
{self._format_list(related_terms)}

Create a codex entry following this structure:

**Overview** (1-3 lines, plain language): What is this?
**Usage** (When player cares): What does it enable or limit?
    When will they encounter it?
**Context** (Neutral background): Surface-level facts that clarify
    without revealing causes or twists
**See also** (2-5 terms): Helpful related entries (prefer breadth over recursion)
**Notes** (Optional): Variants, pronunciation, units, accessibility hints
**Lineage** (Traceability): TU reference

Keep it:
- Player-safe (no spoilers, no internal logic)
- Concise and actionable (not lecturing)
- Accessibility-ready (clear headings, descriptive links)
- Localization-friendly (avoid idioms)

Respond in JSON format:
```json
{{
  "title": "Entry title",
  "slug": "kebab-case-anchor",
  "overview": "Brief overview",
  "usage": "When and why this matters to the player",
  "context": "Neutral background information",
  "see_also": ["Term 1", "Term 2", "Term 3"],
  "notes": "Optional pronunciation, variants, etc.",
  "lineage": "TU reference"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "codex_entry",
                    "entry": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating entry: {e}",
            )

    def _create_crosslinks(self, context: RoleContext) -> RoleResult:
        """Generate See also links between entries."""
        system_prompt = self.build_system_prompt(context)

        entries = context.additional_context.get("entries", [])

        user_prompt = f"""# Task: Create Crosslinks

{self.format_artifacts(context.artifacts)}

## Existing Entries
{self._format_entries(entries)}

Create a crosslink map that:
1. Ensures each entry has 2-5 useful See also links
2. Avoids orphaned entries (entries with no links to them)
3. Creates triangles (A→B→C→A) to improve navigation
4. Prioritizes breadth over deep recursion
5. Uses descriptive link text

Respond in JSON format:
```json
{{
  "crosslinks": [
    {{
      "from": "Entry A",
      "to": ["Entry B", "Entry C", "Entry D"],
      "rationale": "Why these links help"
    }}
  ],
  "orphans": ["Entry X", "Entry Y"],
  "coverage_notes": "What's missing or needs more links"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "crosslink_map",
                    "crosslinks": data.get("crosslinks", []),
                    "orphans": data.get("orphans", []),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating crosslinks: {e}",
            )

    def _check_coverage(self, context: RoleContext) -> RoleResult:
        """Identify missing or incomplete entries."""
        system_prompt = self.build_system_prompt(context)

        manuscript_terms = context.additional_context.get("manuscript_terms", [])
        existing_entries = context.additional_context.get("existing_entries", [])

        user_prompt = f"""# Task: Check Coverage

{self.format_artifacts(context.artifacts)}

## Terms in Manuscript
{self._format_list(manuscript_terms)}

## Existing Codex Entries
{self._format_list(existing_entries)}

Analyze coverage and identify:
1. **Missing Entries**: High-frequency terms without entries
2. **Incomplete Entries**: Entries lacking context or links
3. **Dead Links**: See also references to non-existent entries
4. **Priority Order**: What should be created first (based on player value)

Respond in JSON format:
```json
{{
  "missing": [
    {{
      "term": "Term name",
      "frequency": 15,
      "priority": "high|medium|low",
      "rationale": "Why this matters"
    }}
  ],
  "incomplete": ["Entry 1", "Entry 2"],
  "dead_links": ["Broken Link 1", "Broken Link 2"],
  "recommendations": ["Action 1", "Action 2"]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "coverage_report",
                    "missing": data.get("missing", []),
                    "incomplete": data.get("incomplete", []),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error checking coverage: {e}",
            )

    def _validate_accessibility(self, context: RoleContext) -> RoleResult:
        """Ensure accessibility standards are met."""
        system_prompt = self.build_system_prompt(context)

        entry = context.additional_context.get("entry", {})

        user_prompt = f"""# Task: Validate Accessibility

{self.format_artifacts(context.artifacts)}

## Entry to Validate
{self._format_dict(entry)}

Check for accessibility issues:
1. **Headings**: Are they meaningful and hierarchical?
2. **Links**: Is link text descriptive (not "click here")?
3. **Clarity**: Are sentences concise and actionable?
4. **Structure**: Does it follow Overview → Usage → Context → See also?
5. **Alt Text Notes**: Any images/diagrams that need alt text guidance?

Respond in JSON format:
```json
{{
  "is_accessible": true|false,
  "issues": [
    {{
      "type": "heading|link|clarity|structure|alt",
      "description": "Issue description",
      "suggestion": "How to fix"
    }}
  ]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "accessibility_validation",
                    "is_accessible": data.get("is_accessible", False),
                    "issues": data.get("issues", []),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating accessibility: {e}",
            )

    def _create_glossary_slice(self, context: RoleContext) -> RoleResult:
        """Extract terminology for translation coordination."""
        system_prompt = self.build_system_prompt(context)

        entries = context.additional_context.get("entries", [])

        user_prompt = f"""# Task: Create Glossary Slice

{self.format_artifacts(context.artifacts)}

## Entries
{self._format_entries(entries)}

Create a glossary slice for translation coordination:
1. **Key Terms**: Terms that need consistent translation
2. **Register**: Formal/informal level for each term
3. **Context**: Usage notes for translators
4. **Variants**: Alternative phrasings or related terms
5. **Idioms**: Problematic phrases that need special handling

Respond in JSON format:
```json
{{
  "terms": [
    {{
      "term": "Term",
      "register": "formal|informal|technical",
      "context": "When/how this is used",
      "variants": ["Variant 1", "Variant 2"],
      "translation_notes": "Special considerations"
    }}
  ]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "glossary_slice",
                    "terms": data.get("terms", []),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating glossary: {e}",
            )

    def _format_list(self, items: list[str]) -> str:
        """Format list as bullet list."""
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)

    def _format_entries(self, entries: list[dict[str, Any]]) -> str:
        """Format entries list for prompt."""
        if not entries:
            return "(no entries provided)"

        formatted = []
        for entry in entries:
            title = entry.get("title", "Untitled")
            overview = entry.get("overview", "No overview")
            formatted.append(f"- **{title}**: {overview}")
        return "\n".join(formatted)

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format dictionary as bullet list."""
        if not d:
            return "(empty)"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())
