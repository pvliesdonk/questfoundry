"""
Semantic validation for tool interfaces and schemas.

Checks for semantic ambiguity issues as defined in meta/docs/semantic-conventions.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SemanticIssue:
    """A semantic validation issue."""

    severity: Severity
    location: str  # e.g., "domain-v4/tools/web_search.json:output_schema.status"
    rule: str  # e.g., "banned_field_name"
    message: str
    suggestion: str | None = None


@dataclass
class SemanticValidationResult:
    """Result of semantic validation."""

    issues: list[SemanticIssue] = field(default_factory=list)
    files_checked: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def add(self, issue: SemanticIssue) -> None:
        self.issues.append(issue)


# Banned or discouraged field names
BANNED_FIELDS = {
    # Color codes - should use pass/warn/fail
    "green": "Use 'pass' instead of color code",
    "yellow": "Use 'warn' instead of color code",
    "red": "Use 'fail' instead of color code",
}

DISCOURAGED_FIELDS = {
    # Passive language
    "hint": "Use 'recommended_action' or 'recovery_action' for directive guidance",
    # Ambiguous time values
    "any": "Use specific values like 'all_time' instead of generic 'any'",
    # Conflated status
    "completing": "Use 'in_progress' with percent_complete instead",
}

# Emphatic patterns in tool descriptions that indicate biasing
BIASING_PATTERNS = [
    (r"\bONLY\b", "Emphatic 'ONLY' may override system prompt instructions"),
    (r"\bALWAYS\b", "Emphatic 'ALWAYS' may override system prompt instructions"),
    (r"\bMUST\b", "Emphatic 'MUST' may override system prompt instructions"),
    (r"\bNEVER\b", "Emphatic 'NEVER' may override system prompt instructions"),
    (r"Showrunner", "Role-specific mention biases tool selection"),
    (r"orchestrator[s]?\s+use", "Role-specific guidance biases tool selection"),
    (r"NON-ORCHESTRATOR", "Role-specific exclusion biases tool selection"),
]


def check_tool_schema(
    tool_data: dict[str, Any],
    file_path: str,
) -> list[SemanticIssue]:
    """Check a tool schema for semantic issues."""
    issues: list[SemanticIssue] = []

    tool_id = tool_data.get("id", "unknown")

    # Check description for biasing language
    description = tool_data.get("description", "")
    for pattern, message in BIASING_PATTERNS:
        if re.search(pattern, description):
            issues.append(
                SemanticIssue(
                    severity=Severity.WARNING,
                    location=f"{file_path}:description",
                    rule="biasing_language",
                    message=f"Tool '{tool_id}': {message}",
                    suggestion="Remove role-specific or emphatic language from tool description",
                )
            )

    # Check input schema
    input_schema = tool_data.get("input_schema", {})
    issues.extend(_check_schema_fields(input_schema, f"{file_path}:input_schema", tool_id))

    # Check output schema
    output_schema = tool_data.get("output_schema", {})
    issues.extend(_check_schema_fields(output_schema, f"{file_path}:output_schema", tool_id))

    return issues


def _check_schema_fields(
    schema: dict[str, Any],
    location: str,
    tool_id: str,
) -> list[SemanticIssue]:
    """Recursively check schema fields for banned/discouraged names."""
    issues: list[SemanticIssue] = []

    properties = schema.get("properties", {})
    for field_name, field_def in properties.items():
        # Check for banned field names
        if field_name in BANNED_FIELDS:
            issues.append(
                SemanticIssue(
                    severity=Severity.ERROR,
                    location=f"{location}.{field_name}",
                    rule="banned_field_name",
                    message=f"Tool '{tool_id}': Banned field name '{field_name}'",
                    suggestion=BANNED_FIELDS[field_name],
                )
            )

        # Check for discouraged field names
        if field_name in DISCOURAGED_FIELDS:
            issues.append(
                SemanticIssue(
                    severity=Severity.WARNING,
                    location=f"{location}.{field_name}",
                    rule="discouraged_field_name",
                    message=f"Tool '{tool_id}': Discouraged field name '{field_name}'",
                    suggestion=DISCOURAGED_FIELDS[field_name],
                )
            )

        # Check enum values
        if "enum" in field_def:
            for enum_val in field_def["enum"]:
                if enum_val in BANNED_FIELDS:
                    issues.append(
                        SemanticIssue(
                            severity=Severity.ERROR,
                            location=f"{location}.{field_name}",
                            rule="banned_enum_value",
                            message=f"Tool '{tool_id}': Banned enum value '{enum_val}' in {field_name}",
                            suggestion=BANNED_FIELDS[enum_val],
                        )
                    )
                if enum_val in DISCOURAGED_FIELDS:
                    issues.append(
                        SemanticIssue(
                            severity=Severity.WARNING,
                            location=f"{location}.{field_name}",
                            rule="discouraged_enum_value",
                            message=f"Tool '{tool_id}': Discouraged enum value '{enum_val}' in {field_name}",
                            suggestion=DISCOURAGED_FIELDS[enum_val],
                        )
                    )

        # Recurse into nested objects
        if field_def.get("type") == "object":
            issues.extend(_check_schema_fields(field_def, f"{location}.{field_name}", tool_id))

        # Check array items
        if field_def.get("type") == "array" and "items" in field_def:
            items = field_def["items"]
            if isinstance(items, dict) and items.get("type") == "object":
                issues.extend(_check_schema_fields(items, f"{location}.{field_name}[]", tool_id))

    return issues


def validate_domain_semantics(domain_path: Path) -> SemanticValidationResult:
    """Validate all tools in a domain for semantic issues."""
    import json

    result = SemanticValidationResult()

    tools_dir = domain_path / "tools"
    if not tools_dir.exists():
        return result

    for tool_file in tools_dir.glob("*.json"):
        result.files_checked += 1

        try:
            with open(tool_file) as f:
                tool_data = json.load(f)

            rel_path = str(tool_file.relative_to(domain_path.parent))
            issues = check_tool_schema(tool_data, rel_path)
            result.issues.extend(issues)

        except json.JSONDecodeError as e:
            result.add(
                SemanticIssue(
                    severity=Severity.ERROR,
                    location=str(tool_file),
                    rule="invalid_json",
                    message=f"Invalid JSON: {e}",
                )
            )
        except Exception as e:
            result.add(
                SemanticIssue(
                    severity=Severity.ERROR,
                    location=str(tool_file),
                    rule="file_error",
                    message=f"Error reading file: {e}",
                )
            )

    return result
