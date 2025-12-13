"""Validate artifact tool for v4 runtime.

This tool validates artifacts against their type schemas and quality bars.
It wraps the existing validation.py functionality and adds quality bar
information for LLM-based evaluation.

Quality Bars (8 total):
1. Integrity - No contradictions or dead links
2. Reachability - Critical content accessible
3. Nonlinearity - Branches matter
4. Gateways - Conditions enforceable
5. Style - Voice consistent
6. Determinism - Reproducible where promised
7. Presentation - Spoiler-safe
8. Accessibility - Navigation clear

Schema validation is programmatic. Quality bar evaluation uses LLM rubrics
defined in domain-v4/governance/quality-criteria/*.json.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)

# All 8 quality bars
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


class ValidateArtifactTool(BaseTool):
    """Validate an artifact against its type schema and quality bars.

    Use this tool to check if an artifact is valid before persisting or
    submitting for review. Returns detailed validation results with
    specific errors and suggested fixes.

    Validation modes:
    - schema_only: Check against Pydantic schema (fast, programmatic)
    - bars_only: Check quality criteria applicability (informational)
    - full: Both schema validation and quality bar information
    """

    name: str = "validate_artifact"
    description: str = (
        "Validate an artifact against its type schema and quality bars. "
        "Use validation_mode='schema_only' for quick schema checks, "
        "'bars_only' to see which quality bars apply, or 'full' for both. "
        "Returns validation results with errors and suggested fixes."
    )

    # Injected by registry
    studio: Any = Field(default=None, exclude=True)

    def _run(
        self,
        artifact_type_id: str,
        artifact_id: str | None = None,
        artifact_data: dict[str, Any] | None = None,
        validation_mode: str = "full",
        bars_to_check: list[str] | None = None,
    ) -> str:
        """Validate an artifact.

        Parameters
        ----------
        artifact_type_id : str
            Type of the artifact (e.g., 'section', 'hook_card')
        artifact_id : str | None
            ID of the artifact to validate (if already persisted)
        artifact_data : dict | None
            The artifact content to validate (if not already persisted)
        validation_mode : str
            What to validate: 'schema_only', 'bars_only', or 'full'
        bars_to_check : list[str] | None
            Specific bars to check (default: all 8)

        Returns
        -------
        str
            JSON result with validation status and details
        """
        result: dict[str, Any] = {
            "valid": True,
            "artifact_type": artifact_type_id,
            "schema_errors": [],
            "bar_results": [],
        }

        # Validate mode
        if validation_mode not in ("schema_only", "bars_only", "full"):
            return json.dumps({
                "valid": False,
                "error": f"Invalid validation_mode: {validation_mode}. "
                "Use 'schema_only', 'bars_only', or 'full'.",
            })

        # Need either artifact_id or artifact_data
        if artifact_data is None and artifact_id is None:
            return json.dumps({
                "valid": False,
                "error": "Provide either artifact_id (for persisted) or artifact_data (for new).",
            })

        # Schema validation
        if validation_mode in ("schema_only", "full"):
            schema_result = self._validate_schema(artifact_type_id, artifact_data or {})
            if not schema_result.get("success", False):
                result["valid"] = False
                result["schema_errors"] = self._format_schema_errors(schema_result)

        # Quality bar information
        if validation_mode in ("bars_only", "full"):
            bars = bars_to_check or QUALITY_BARS
            # Validate bar names
            invalid_bars = [b for b in bars if b not in QUALITY_BARS]
            if invalid_bars:
                return json.dumps({
                    "valid": False,
                    "error": f"Invalid quality bars: {invalid_bars}. "
                    f"Valid bars: {QUALITY_BARS}",
                })

            bar_results = self._get_bar_applicability(artifact_type_id, bars)
            result["bar_results"] = bar_results

        return json.dumps(result, indent=2)

    def _validate_schema(
        self, artifact_type: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate artifact data against its schema."""
        try:
            from questfoundry.runtime.validation import validate_artifact

            return validate_artifact(artifact_type, data)
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _format_schema_errors(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Format schema validation errors into the output format."""
        errors: list[dict[str, Any]] = []

        # Handle missing fields
        for field in result.get("missing_fields", []):
            errors.append({
                "field": field,
                "error": "Required field is missing",
                "suggested_fix": f"Add the '{field}' field with a valid value",
            })

        # Handle invalid fields
        for field_info in result.get("invalid_fields", []):
            if isinstance(field_info, dict):
                errors.append({
                    "field": field_info.get("field", "unknown"),
                    "error": field_info.get("error", "Invalid value"),
                    "suggested_fix": field_info.get("hint", "Check field type and constraints"),
                })
            else:
                errors.append({
                    "field": str(field_info),
                    "error": "Invalid value",
                    "suggested_fix": "Check field type and constraints",
                })

        # Add hint if available
        if result.get("hint"):
            errors.append({
                "field": "_general",
                "error": "Validation hint",
                "suggested_fix": result["hint"],
            })

        return errors

    def _get_bar_applicability(
        self, artifact_type: str, bars: list[str]
    ) -> list[dict[str, Any]]:
        """Get quality bar applicability for an artifact type.

        This returns which bars apply to this artifact type and their
        evaluation requirements. Actual bar evaluation requires LLM
        judgment using the rubrics from domain-v4/governance/quality-criteria/.
        """
        bar_results: list[dict[str, Any]] = []

        # Load quality criteria from studio if available
        quality_criteria = {}
        if self.studio and hasattr(self.studio, "quality_criteria"):
            quality_criteria = self.studio.quality_criteria

        for bar in bars:
            bar_info: dict[str, Any] = {
                "bar": bar,
                "status": "pending",  # pending = not evaluated yet
                "evidence": "",
                "smallest_fix": "",
            }

            # Check if we have criteria definition
            criteria = quality_criteria.get(bar)
            if criteria:
                # Check if this bar applies to this artifact type
                applies_to = getattr(criteria, "applies_to", None)
                if applies_to:
                    artifact_types = applies_to.get("artifact_types", [])
                    if artifact_types and artifact_type not in artifact_types:
                        bar_info["status"] = "not_applicable"
                        bar_info["evidence"] = (
                            f"Bar '{bar}' does not apply to '{artifact_type}'. "
                            f"Applies to: {artifact_types}"
                        )
                    else:
                        bar_info["status"] = "requires_evaluation"
                        bar_info["evidence"] = (
                            f"Bar '{bar}' applies. Evaluation requires LLM judgment "
                            f"using the rubric defined in quality-criteria/{bar}.json"
                        )
                        if hasattr(criteria, "failure_guidance"):
                            bar_info["smallest_fix"] = criteria.failure_guidance
                else:
                    # No applies_to restriction - applies to all
                    bar_info["status"] = "requires_evaluation"
                    bar_info["evidence"] = (
                        f"Bar '{bar}' applies to all artifact types. "
                        "Evaluation requires LLM judgment."
                    )
            else:
                # No criteria loaded - provide generic info
                bar_info["status"] = "unknown"
                bar_info["evidence"] = (
                    f"Quality criteria for '{bar}' not loaded. "
                    "Check domain-v4/governance/quality-criteria/"
                )

            bar_results.append(bar_info)

        return bar_results
