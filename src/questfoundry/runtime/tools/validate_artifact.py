"""
Validate Artifact tool implementation.

Validates artifacts against their type schema and the 8 quality bars:
1. Integrity - No contradictions or dead links
2. Reachability - Critical content accessible
3. Nonlinearity - Branches matter
4. Gateways - Conditions enforceable
5. Style - Voice consistent
6. Determinism - Reproducible where promised
7. Presentation - Spoiler-safe
8. Accessibility - Navigation clear
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    from questfoundry.runtime.models import ArtifactType


class BarStatus(str, Enum):
    """Status of a quality bar check."""

    GREEN = "green"  # Passes
    YELLOW = "yellow"  # Minor issues, can proceed
    RED = "red"  # Fails, needs fix


@dataclass
class SchemaError:
    """A schema validation error."""

    field: str
    error: str
    suggested_fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "error": self.error,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class BarResult:
    """Result of a quality bar check."""

    bar: str
    status: BarStatus
    evidence: str
    smallest_fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar": self.bar,
            "status": self.status.value,
            "evidence": self.evidence,
            "smallest_fix": self.smallest_fix,
        }


# The 8 Quality Bars
QUALITY_BARS = [
    "integrity",
    "reachability",
    "nonlinearity",
    "gateways",
    "style",
    "determinism",
    "presentation",
    "accessibility",
]


@register_tool("validate_artifact")
class ValidateArtifactTool(BaseTool):
    """
    Validate an artifact against its type schema and quality bars.

    Returns structured validation results with specific errors and fixes.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute validation."""
        artifact_id = args.get("artifact_id")
        artifact_type_id = args.get("artifact_type_id")
        artifact_data = args.get("artifact_data")
        validation_mode = args.get("validation_mode", "full")
        bars_to_check = args.get("bars_to_check", QUALITY_BARS)

        # Get artifact data - either from args or from storage
        if artifact_data is None and artifact_id and self._context.project:
            artifact_data = self._context.project.get_artifact(artifact_id)
            if artifact_data:
                # Extract user fields
                artifact_data = {k: v for k, v in artifact_data.items() if not k.startswith("_")}

        if artifact_data is None:
            return ToolResult(
                success=False,
                data={"valid": False},
                error="No artifact data provided or found",
            )

        # Get artifact type definition
        artifact_type = None
        for at in self._context.studio.artifact_types:
            if at.id == artifact_type_id:
                artifact_type = at
                break

        if not artifact_type:
            return ToolResult(
                success=False,
                data={"valid": False},
                error=f"Artifact type not found: {artifact_type_id}",
            )

        schema_errors: list[SchemaError] = []
        bar_results: list[BarResult] = []

        # Schema validation
        if validation_mode in ("schema_only", "full"):
            schema_errors = self._validate_schema(artifact_type, artifact_data)

        # Quality bar validation
        if validation_mode in ("bars_only", "full"):
            bar_results = self._validate_bars(artifact_type, artifact_data, bars_to_check)

        # Determine overall validity
        has_schema_errors = len(schema_errors) > 0
        has_red_bars = any(br.status == BarStatus.RED for br in bar_results)
        valid = not has_schema_errors and not has_red_bars

        return ToolResult(
            success=True,
            data={
                "valid": valid,
                "schema_errors": [e.to_dict() for e in schema_errors],
                "bar_results": [br.to_dict() for br in bar_results],
            },
        )

    def _validate_schema(
        self, artifact_type: ArtifactType, artifact_data: dict[str, Any]
    ) -> list[SchemaError]:
        """Validate artifact against its schema."""
        errors: list[SchemaError] = []

        if not artifact_type.fields:
            return errors

        # Check required fields
        for field in artifact_type.fields:
            if field.required and field.name not in artifact_data:
                errors.append(
                    SchemaError(
                        field=field.name,
                        error=f"Required field '{field.name}' is missing",
                        suggested_fix=f"Add field '{field.name}' with type {field.type.value if field.type else 'string'}",
                    )
                )

        # Check field types
        for field in artifact_type.fields:
            if field.name in artifact_data:
                value = artifact_data[field.name]
                field_type = field.type.value if field.type else "string"

                type_error = self._check_field_type(field.name, value, field_type)
                if type_error:
                    errors.append(type_error)

        # Check validation rules
        if artifact_type.validation:
            # Required together
            for group in artifact_type.validation.required_together:
                present = [f for f in group if f in artifact_data]
                if 0 < len(present) < len(group):
                    missing = [f for f in group if f not in artifact_data]
                    errors.append(
                        SchemaError(
                            field=", ".join(missing),
                            error=f"Fields {group} must be provided together",
                            suggested_fix=f"Add missing fields: {missing}",
                        )
                    )

            # Mutually exclusive
            for group in artifact_type.validation.mutually_exclusive:
                present = [f for f in group if f in artifact_data]
                if len(present) > 1:
                    errors.append(
                        SchemaError(
                            field=", ".join(present),
                            error=f"Fields {group} are mutually exclusive",
                            suggested_fix=f"Keep only one of: {present}",
                        )
                    )

        return errors

    def _check_field_type(
        self, field_id: str, value: Any, expected_type: str
    ) -> SchemaError | None:
        """Check if a field value matches the expected type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "text": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "reference": lambda v: isinstance(v, str),
            "enum": lambda v: isinstance(v, str),
        }

        checker = type_checks.get(expected_type)
        if checker and not checker(value):
            return SchemaError(
                field=field_id,
                error=f"Field '{field_id}' has wrong type. Expected {expected_type}, got {type(value).__name__}",
                suggested_fix=f"Convert '{field_id}' to {expected_type}",
            )

        return None

    def _validate_bars(
        self,
        artifact_type: ArtifactType,
        artifact_data: dict[str, Any],
        bars_to_check: list[str],
    ) -> list[BarResult]:
        """Validate against quality bars."""
        results: list[BarResult] = []

        for bar in bars_to_check:
            if bar not in QUALITY_BARS:
                continue

            # Dispatch to bar-specific validation
            validator = getattr(self, f"_check_{bar}", None)
            if validator:
                result = validator(artifact_type, artifact_data)
            else:
                # Default: pass with note that full validation not implemented
                result = BarResult(
                    bar=bar,
                    status=BarStatus.YELLOW,
                    evidence=f"Full {bar} validation requires LLM analysis",
                    smallest_fix=None,
                )

            results.append(result)

        return results

    # ==========================================================================
    # Quality Bar Validators
    # NOTE: Full implementation of these bars requires LLM analysis.
    # These are structural checks that can be done programmatically.
    # ==========================================================================

    def _check_integrity(
        self, artifact_type: ArtifactType, artifact_data: dict[str, Any]
    ) -> BarResult:
        """
        Check integrity - no contradictions or dead links.

        Programmatic checks:
        - References to other artifacts exist
        - No self-contradicting field values
        """
        # Check for reference fields that might be broken
        issues = []

        for field in artifact_type.fields if artifact_type.fields else []:
            if field.type and field.type.value == "reference":
                ref_value = artifact_data.get(field.name)
                if ref_value and self._context.project:
                    # Check if referenced artifact exists
                    ref_artifact = self._context.project.get_artifact(ref_value)
                    if ref_artifact is None:
                        issues.append(
                            f"Reference '{field.name}' points to missing artifact: {ref_value}"
                        )

        if issues:
            return BarResult(
                bar="integrity",
                status=BarStatus.RED,
                evidence="; ".join(issues),
                smallest_fix="Fix broken references or create missing artifacts",
            )

        return BarResult(
            bar="integrity",
            status=BarStatus.GREEN,
            evidence="All references valid (structural check only)",
        )

    def _check_reachability(self, _artifact_type: Any, _artifact_data: dict[str, Any]) -> BarResult:
        """
        Check reachability - critical content accessible.

        TODO: Full implementation requires graph analysis of navigation paths.
        """
        # This requires understanding the story graph - defer to LLM
        return BarResult(
            bar="reachability",
            status=BarStatus.YELLOW,
            evidence="Reachability analysis requires story graph context (LLM analysis)",
            smallest_fix=None,
        )

    def _check_nonlinearity(self, _artifact_type: Any, artifact_data: dict[str, Any]) -> BarResult:
        """
        Check nonlinearity - branches matter.

        Programmatic check: If has choices, at least 2 should be meaningfully different.
        """
        # Check for choices field (common in section artifacts)
        choices = artifact_data.get("choices", [])

        if not choices:
            return BarResult(
                bar="nonlinearity",
                status=BarStatus.GREEN,
                evidence="No choices defined (may be terminal or linear by design)",
            )

        if len(choices) < 2:
            return BarResult(
                bar="nonlinearity",
                status=BarStatus.YELLOW,
                evidence=f"Only {len(choices)} choice(s) - limited branching",
                smallest_fix="Consider adding alternative choices for player agency",
            )

        return BarResult(
            bar="nonlinearity",
            status=BarStatus.GREEN,
            evidence=f"Has {len(choices)} choices (content differentiation requires LLM)",
        )

    def _check_gateways(self, _artifact_type: Any, artifact_data: dict[str, Any]) -> BarResult:
        """
        Check gateways - conditions enforceable.

        Programmatic check: Conditions reference valid state/items.
        """
        # Look for condition/gate/prerequisite fields
        condition_fields = ["condition", "conditions", "prerequisites", "gates", "requires"]

        for field_name in condition_fields:
            if field_name in artifact_data:
                # Has conditions - structural check passes
                return BarResult(
                    bar="gateways",
                    status=BarStatus.GREEN,
                    evidence=f"Has {field_name} defined (enforcement logic requires runtime)",
                )

        return BarResult(
            bar="gateways",
            status=BarStatus.GREEN,
            evidence="No gateway conditions defined",
        )

    def _check_style(self, _artifact_type: Any, _artifact_data: dict[str, Any]) -> BarResult:
        """
        Check style - voice consistent.

        TODO: Full implementation requires LLM analysis of prose.
        """
        return BarResult(
            bar="style",
            status=BarStatus.YELLOW,
            evidence="Style consistency requires LLM analysis",
            smallest_fix=None,
        )

    def _check_determinism(self, _artifact_type: Any, artifact_data: dict[str, Any]) -> BarResult:
        """
        Check determinism - reproducible where promised.

        Programmatic check: Random elements are properly seeded/documented.
        """
        # Look for randomness indicators
        random_indicators = ["random", "dice", "chance", "probability", "roll"]
        text_fields = [v for v in artifact_data.values() if isinstance(v, str) and len(v) > 10]

        has_random = any(
            indicator in text.lower() for text in text_fields for indicator in random_indicators
        )

        if has_random:
            return BarResult(
                bar="determinism",
                status=BarStatus.YELLOW,
                evidence="Content mentions randomness - verify it's intentional and documented",
                smallest_fix="Document random mechanics clearly for players",
            )

        return BarResult(
            bar="determinism",
            status=BarStatus.GREEN,
            evidence="No random elements detected",
        )

    def _check_presentation(self, _artifact_type: Any, _artifact_data: dict[str, Any]) -> BarResult:
        """
        Check presentation - spoiler-safe.

        TODO: Full implementation requires LLM analysis of content flow.
        """
        return BarResult(
            bar="presentation",
            status=BarStatus.YELLOW,
            evidence="Spoiler safety requires LLM analysis of content flow",
            smallest_fix=None,
        )

    def _check_accessibility(self, _artifact_type: Any, artifact_data: dict[str, Any]) -> BarResult:
        """
        Check accessibility - navigation clear.

        Programmatic check: Choices have clear text, not empty/terse.
        """
        choices = artifact_data.get("choices", [])

        if not choices:
            return BarResult(
                bar="accessibility",
                status=BarStatus.GREEN,
                evidence="No navigation choices to evaluate",
            )

        issues = []
        for i, choice in enumerate(choices):
            choice_text = choice.get("text", "") if isinstance(choice, dict) else str(choice)
            if len(choice_text) < 5:
                issues.append(f"Choice {i + 1} text too short: '{choice_text}'")

        if issues:
            return BarResult(
                bar="accessibility",
                status=BarStatus.YELLOW,
                evidence="; ".join(issues),
                smallest_fix="Add clearer, more descriptive choice text",
            )

        return BarResult(
            bar="accessibility",
            status=BarStatus.GREEN,
            evidence=f"All {len(choices)} choices have adequate text",
        )
