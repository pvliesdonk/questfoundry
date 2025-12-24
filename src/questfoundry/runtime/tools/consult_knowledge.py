"""
Consult Knowledge tool implementation.

Retrieves full knowledge content from the studio knowledge base.
Implements the menu+consult pattern from meta/docs/knowledge-patterns.md.
"""

from __future__ import annotations

import re
from typing import Any

from questfoundry.runtime.agent.content_utils import extract_knowledge_content
from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

# Pre-compiled regex for markdown header extraction
_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


@register_tool("consult_knowledge")
class ConsultKnowledgeTool(BaseTool):
    """
    Retrieve detailed guidance from the studio knowledge base.

    Use this when you need full content for entries shown in your Knowledge Menu.
    The Knowledge Menu shows summaries; this tool retrieves complete content
    including examples, procedures, and detailed guidance.

    Common uses:
    - Get detailed examples when summary is insufficient
    - Retrieve full procedures or checklists
    - Access domain-specific guidance for specialized tasks
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute knowledge lookup."""
        entry_id = args.get("entry_id")
        section = args.get("section")

        if not entry_id:
            # List available knowledge entries
            available = list(self._context.studio.knowledge.keys())
            return ToolResult(
                success=True,
                data={
                    "action_outcome": f"listed {len(available)} knowledge entries",
                    "message": (
                        "You called consult_knowledge without specifying an entry_id. "
                        "Available knowledge entries:"
                    ),
                    "available_entries": available,
                    "hint": "Specify entry_id from your Knowledge Menu to get full content.",
                },
            )

        # Find entry in studio knowledge
        entry = self._context.studio.knowledge.get(entry_id)

        if not entry:
            available = list(self._context.studio.knowledge.keys())
            return ToolResult(
                success=False,
                data={"available_entries": available},
                error=f"Knowledge entry not found: {entry_id}",
            )

        # Extract content
        content_text = self._extract_content(entry)

        # If section requested, extract just that section
        if section and content_text:
            section_text = self._extract_section(content_text, section)
            if section_text:
                content_text = section_text
            else:
                # Section not found - return full content with warning
                content_len = len(content_text) if content_text else 0
                return ToolResult(
                    success=True,
                    data={
                        "action_outcome": f"retrieved '{entry_id}' ({content_len} chars)",
                        "entry_id": entry_id,
                        "name": entry.name or entry_id,
                        "layer": entry.layer.value
                        if hasattr(entry.layer, "value")
                        else str(entry.layer),
                        "content": content_text,
                        "related_entries": entry.related_entries or [],
                        "warning": f"Section '{section}' not found. Returning full content.",
                    },
                )

        # Build result
        content_len = len(content_text) if content_text else 0
        result_data: dict[str, Any] = {
            "action_outcome": f"retrieved '{entry_id}' ({content_len} chars)",
            "entry_id": entry_id,
            "name": entry.name or entry_id,
            "layer": entry.layer.value if hasattr(entry.layer, "value") else str(entry.layer),
            "content": content_text or "(No content available)",
            "related_entries": entry.related_entries or [],
        }

        return ToolResult(success=True, data=result_data)

    def _extract_content(self, entry: Any) -> str | None:
        """Extract text content from a knowledge entry.

        Delegates to shared content_utils for consistent handling across
        the codebase. Supports structured, file_ref, and corpus content types.
        """
        # Get domain path from context if available
        domain_path = getattr(self._context, "domain_path", None)
        return extract_knowledge_content(entry, domain_path)

    def _extract_section(self, content: str, section_name: str) -> str | None:
        """Extract a specific section from markdown content.

        Looks for headers matching the section name and returns content
        until the next header of the same or higher level.
        """
        lines = content.split("\n")
        in_section = False
        section_level = 0
        section_lines: list[str] = []

        for line in lines:
            match = _HEADER_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                if in_section:
                    # Check if we've hit a same-level or higher header
                    if level <= section_level:
                        break
                    section_lines.append(line)
                elif title.lower() == section_name.lower():
                    # Found the section
                    in_section = True
                    section_level = level
                    section_lines.append(line)
            elif in_section:
                section_lines.append(line)

        if section_lines:
            return "\n".join(section_lines).strip()
        return None

    def validate_input(self, args: dict[str, Any]) -> None:
        """Validate input arguments."""
        super().validate_input(args)

        entry_id = args.get("entry_id")
        if entry_id is not None and not isinstance(entry_id, str):
            raise ToolValidationError("entry_id must be a string")

        section = args.get("section")
        if section is not None and not isinstance(section, str):
            raise ToolValidationError("section must be a string")
