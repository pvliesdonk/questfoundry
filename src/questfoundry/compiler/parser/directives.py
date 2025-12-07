"""MyST directive parser for extracting structured data from domain files.

This parser extracts custom directives from MyST markdown files. Directives use
the MyST fence syntax with YAML content:

    :::{directive-name}
    key: value
    nested:
      - item1
      - item2
    :::

The parser extracts all directives from a file and returns them as structured
Directive objects with validated YAML content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DirectiveType(Enum):
    """All supported MyST directive types."""

    # Role directives
    ROLE_META = "role-meta"
    ROLE_TOOLS = "role-tools"
    ROLE_CONSTRAINTS = "role-constraints"
    ROLE_PROMPT = "role-prompt"

    # Loop directives
    LOOP_META = "loop-meta"
    GRAPH_NODE = "graph-node"
    GRAPH_EDGE = "graph-edge"
    QUALITY_GATE = "quality-gate"

    # Ontology directives
    ARTIFACT_TYPE = "artifact-type"
    ARTIFACT_FIELD = "artifact-field"
    ENUM_TYPE = "enum-type"
    ENUM_VALUE = "enum-value"

    # Protocol directives
    INTENT_TYPE = "intent-type"
    ROUTING_RULE = "routing-rule"
    QUALITY_BAR = "quality-bar"

    @classmethod
    def from_string(cls, name: str) -> DirectiveType | None:
        """Convert a directive name string to DirectiveType enum."""
        for directive_type in cls:
            if directive_type.value == name:
                return directive_type
        return None


@dataclass
class Directive:
    """A parsed MyST directive with its content."""

    type: DirectiveType
    content: dict[str, Any]
    line_number: int
    raw_content: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the directive content."""
        return self.content.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a value from the directive content."""
        return self.content[key]

    def keys(self) -> list[str]:
        """Get all keys in the directive content."""
        return list(self.content.keys())


@dataclass
class ParseResult:
    """Result of parsing a MyST file."""

    directives: list[Directive] = field(default_factory=list)
    prose_sections: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    source_path: Path | None = None

    def get_directives(self, directive_type: DirectiveType) -> list[Directive]:
        """Get all directives of a specific type."""
        return [d for d in self.directives if d.type == directive_type]

    def get_directive(self, directive_type: DirectiveType) -> Directive | None:
        """Get the first directive of a specific type (for singular directives)."""
        directives = self.get_directives(directive_type)
        return directives[0] if directives else None

    @property
    def has_errors(self) -> bool:
        """Check if parsing produced any errors."""
        return len(self.errors) > 0


# Regex pattern for MyST directive blocks
# Matches: :::{directive-name}\n...content...\n:::
DIRECTIVE_PATTERN = re.compile(
    r"^:::\{([a-z-]+)\}\s*$\n"  # Opening fence with directive name
    r"(.*?)"  # Content (non-greedy)
    r"^:::\s*$",  # Closing fence
    re.MULTILINE | re.DOTALL,
)

# Pattern for markdown headers
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_myst_string(content: str, source_path: Path | None = None) -> ParseResult:
    """Parse MyST content string and extract all directives.

    Args:
        content: The MyST markdown content to parse.
        source_path: Optional path to the source file (for error messages).

    Returns:
        ParseResult containing all extracted directives and any errors.
    """
    result = ParseResult(source_path=source_path)

    # Find all directive blocks
    for match in DIRECTIVE_PATTERN.finditer(content):
        directive_name = match.group(1)
        directive_content = match.group(2).strip()

        # Calculate line number
        line_number = content[: match.start()].count("\n") + 1

        # Try to parse as DirectiveType
        directive_type = DirectiveType.from_string(directive_name)

        if directive_type is None:
            result.errors.append(
                f"Unknown directive '{directive_name}' at line {line_number}"
            )
            continue

        # Parse YAML content
        try:
            # Handle special case for role-prompt which may contain Jinja2 templates
            if directive_type == DirectiveType.ROLE_PROMPT:
                # Store as raw string under 'template' key
                parsed_content = {"template": directive_content}
            else:
                parsed_content = yaml.safe_load(directive_content) or {}

            # Ensure we have a dict
            if not isinstance(parsed_content, dict):
                # If it's a list (like role-tools), wrap it
                if isinstance(parsed_content, list):
                    parsed_content = {"items": parsed_content}
                else:
                    parsed_content = {"value": parsed_content}

        except yaml.YAMLError as e:
            result.errors.append(
                f"Invalid YAML in '{directive_name}' at line {line_number}: {e}"
            )
            continue

        directive = Directive(
            type=directive_type,
            content=parsed_content,
            line_number=line_number,
            raw_content=directive_content,
        )
        result.directives.append(directive)

    # Extract prose sections (headers and their content)
    # This is useful for documentation extraction
    current_header: str | None = None
    current_content: list[str] = []

    # Remove directive blocks from content for prose extraction
    prose_content = DIRECTIVE_PATTERN.sub("", content)

    for line in prose_content.split("\n"):
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            # Save previous section
            if current_header is not None:
                result.prose_sections[current_header] = "\n".join(current_content).strip()

            current_header = header_match.group(2).strip()
            current_content = []
        elif current_header is not None:
            current_content.append(line)

    # Save last section
    if current_header is not None:
        result.prose_sections[current_header] = "\n".join(current_content).strip()

    return result


def parse_myst_file(path: Path) -> ParseResult:
    """Parse a MyST file and extract all directives.

    Args:
        path: Path to the MyST markdown file.

    Returns:
        ParseResult containing all extracted directives and any errors.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        result = ParseResult(source_path=path)
        result.errors.append(f"Failed to read file: {e}")
        return result

    return parse_myst_string(content, source_path=path)


def parse_domain_directory(domain_path: Path) -> dict[str, list[ParseResult]]:
    """Parse all MyST files in the domain directory structure.

    Args:
        domain_path: Path to the domain directory.

    Returns:
        Dictionary mapping subdirectory names to lists of ParseResults.
    """
    results: dict[str, list[ParseResult]] = {
        "roles": [],
        "loops": [],
        "ontology": [],
        "protocol": [],
    }

    for subdir in results:
        subdir_path = domain_path / subdir
        if not subdir_path.exists():
            continue

        for md_file in subdir_path.glob("*.md"):
            # Skip non-domain files like ARCHITECTURE.md
            if md_file.name.startswith("_") or md_file.name.isupper():
                continue

            result = parse_myst_file(md_file)
            results[subdir].append(result)

    return results
