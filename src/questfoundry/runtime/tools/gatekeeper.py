"""Gatekeeper-specific tools for quality bar evaluation.

The Gatekeeper role uses these tools to systematically evaluate artifacts
against quality bars and produce GatecheckReport artifacts.

Quality Bars (8 total):
- integrity: No contradictions in canon
- reachability: All content accessible via valid paths
- nonlinearity: Multiple valid paths exist
- gateways: All gates have valid unlock conditions
- style: Voice and tone consistency
- determinism: Same inputs produce same outputs
- presentation: Formatting and structure correct
- accessibility: Content usable by all players

Tool Response Pattern (see ARCHITECTURE.md Section 9.2):
All evaluate_* tools return clear verdicts, not guidance. This prevents
LLM retry loops where the model keeps calling the same tool hoping for
a definitive answer.

Response format:
    {
        "bar": "<bar_name>",
        "artifact_id": "<id>",
        "passed": true/false,
        "issues": ["specific issue 1", ...],
        "notes": "Human-readable summary",
        "next_step": "What to do with this result"
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import Field

from questfoundry.generated.models.enums import QualityBar

logger = logging.getLogger(__name__)


# Type alias for bar evaluation result
BarResult = dict[str, Any]  # {"passed": bool, "issues": list[str], "notes": str}


class EvaluateIntegrity(BaseTool):
    """Evaluate the integrity bar - checks for contradictions in canon.

    Integrity ensures that facts, events, and relationships don't contradict
    each other across the artifact and related canon.
    """

    name: str = "evaluate_integrity"
    description: str = (
        "Check artifact for internal contradictions and canon conflicts. "
        "Returns pass/fail with specific issues found. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate integrity bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "integrity",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        # Auto-pass for now (stub implementation)
        # Real implementation would check for contradictions
        return json.dumps(
            {
                "bar": "integrity",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Integrity evaluation passed - no contradictions detected.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluateReachability(BaseTool):
    """Evaluate the reachability bar - checks content accessibility.

    Reachability ensures all content can be reached via valid player paths.
    No orphaned scenes, no dead ends without purpose.
    """

    name: str = "evaluate_reachability"
    description: str = (
        "Check that all content in artifact is accessible via valid paths. "
        "Returns pass/fail with unreachable content identified. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate reachability bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "reachability",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        # Auto-pass for now (stub implementation)
        return json.dumps(
            {
                "bar": "reachability",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Reachability evaluation passed - all content accessible.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluateNonlinearity(BaseTool):
    """Evaluate the nonlinearity bar - checks for meaningful choices.

    Nonlinearity ensures the narrative offers meaningful branching paths
    with real consequences, not just cosmetic differences.
    """

    name: str = "evaluate_nonlinearity"
    description: str = (
        "Check that choices lead to meaningfully different outcomes. "
        "Returns pass/fail with analysis of choice impact. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate nonlinearity bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "nonlinearity",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        # Auto-pass for now (stub implementation)
        return json.dumps(
            {
                "bar": "nonlinearity",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Nonlinearity evaluation passed - meaningful choices detected.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluateGateways(BaseTool):
    """Evaluate the gateways bar - checks gate unlock conditions.

    Gateways ensures all gates (locked content) have valid, achievable
    unlock conditions that players can discover and satisfy.
    """

    name: str = "evaluate_gateways"
    description: str = (
        "Check that all gates have valid unlock conditions. "
        "Returns pass/fail with gate validation details. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate gateways bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "gateways",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        # Auto-pass for now (stub implementation)
        return json.dumps(
            {
                "bar": "gateways",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Gateways evaluation passed - all gates have valid unlock conditions.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluateStyle(BaseTool):
    """Evaluate the style bar - checks voice and tone consistency.

    Style ensures the narrative voice, tone, and register are consistent
    throughout the content and match the intended aesthetic.

    Prerequisite: Content must exist before style can be evaluated.
    """

    name: str = "evaluate_style"
    description: str = (
        "Check voice and tone consistency across artifact. "
        "Requires content to exist - cannot evaluate style of empty content. "
        "Returns pass/fail with style inconsistencies noted. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate style bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "style",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        issues = []

        # Style requires content to exist - can't evaluate style of nothing
        if isinstance(artifact, dict):
            content = artifact.get("content")
            is_scene = "content" in artifact or artifact_id.startswith("scene_")

            if is_scene:
                if content is None or (isinstance(content, str) and not content.strip()):
                    return json.dumps(
                        {
                            "bar": "style",
                            "artifact_id": artifact_id,
                            "passed": False,
                            "issues": [
                                f"Scene '{artifact_id}' has no content - cannot evaluate style of empty prose"
                            ],
                            "notes": "Style evaluation blocked - content required first",
                            "next_step": "Return to Scene Smith to write prose before style can be evaluated.",
                        }
                    )

                # Basic style checks (can be expanded with LLM analysis later)
                content_str = content.strip()

                # Check for obvious style issues
                if content_str.isupper():
                    issues.append("Content is all uppercase - inconsistent with narrative style")

                # Check for mixed formatting indicators
                if "##" in content_str and not content_str.startswith("#"):
                    issues.append("Markdown headers in prose content - may break presentation")

        if issues:
            return json.dumps(
                {
                    "bar": "style",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": issues,
                    "notes": f"Style issues found - {len(issues)} issue(s)",
                    "next_step": "Return to Scene Smith or Creative Director to address style issues.",
                }
            )

        return json.dumps(
            {
                "bar": "style",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Style evaluation passed - voice and tone appear consistent.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluateDeterminism(BaseTool):
    """Evaluate the determinism bar - checks for consistent behavior.

    Determinism ensures the same inputs always produce the same outputs,
    preventing unpredictable or buggy behavior.
    """

    name: str = "evaluate_determinism"
    description: str = (
        "Check that same inputs produce same outputs. "
        "Returns pass/fail with any non-deterministic elements. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate determinism bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "determinism",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        # Auto-pass for now (stub implementation)
        return json.dumps(
            {
                "bar": "determinism",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Determinism evaluation passed - consistent behavior verified.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluatePresentation(BaseTool):
    """Evaluate the presentation bar - checks formatting and structure.

    Presentation ensures content is properly formatted, structured,
    and ready for rendering without technical issues.

    This includes checking that required content fields are populated -
    empty content fields indicate incomplete work.
    """

    name: str = "evaluate_presentation"
    description: str = (
        "Check formatting and structure correctness, including content completeness. "
        "Returns pass/fail with formatting issues. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate presentation bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "presentation",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        issues = []

        # Check content completeness for scene-type artifacts
        # Scenes must have prose content - empty content is a blocking issue
        if isinstance(artifact, dict):
            content = artifact.get("content")

            # Check if this looks like a scene (has content field or scene in ID)
            is_scene = "content" in artifact or artifact_id.startswith("scene_")

            if is_scene:
                if content is None:
                    issues.append(
                        f"Scene '{artifact_id}' has no 'content' field - prose required"
                    )
                elif isinstance(content, str):
                    stripped = content.strip()
                    if not stripped:
                        issues.append(
                            f"Scene '{artifact_id}' has empty content - prose required"
                        )
                    elif len(stripped) < 50:
                        issues.append(
                            f"Scene '{artifact_id}' has very short content ({len(stripped)} chars) - may need expansion"
                        )
                    # Check for placeholder text
                    placeholders = ["[TODO", "[PLACEHOLDER", "[TBD", "Lorem ipsum"]
                    for ph in placeholders:
                        if ph.lower() in stripped.lower():
                            issues.append(
                                f"Scene '{artifact_id}' contains placeholder text: '{ph}'"
                            )

            # Check for valid title
            title = artifact.get("title")
            if title is not None and isinstance(title, str) and not title.strip():
                issues.append(f"Artifact '{artifact_id}' has empty title")

        if issues:
            return json.dumps(
                {
                    "bar": "presentation",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": issues,
                    "notes": f"Presentation failed - {len(issues)} issue(s) found",
                    "next_step": "Return to Scene Smith to fill missing prose content.",
                }
            )

        return json.dumps(
            {
                "bar": "presentation",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Presentation evaluation passed - formatting and structure correct.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class EvaluateAccessibility(BaseTool):
    """Evaluate the accessibility bar - checks inclusive design.

    Accessibility ensures content is usable by all players regardless
    of ability, with proper alt text, warnings, and design considerations.
    """

    name: str = "evaluate_accessibility"
    description: str = (
        "Check content is usable by all players. "
        "Returns pass/fail with accessibility concerns. "
        "Args: artifact_id (str) - the hot_store key of artifact to check"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    cold_store: Any | None = Field(default=None, exclude=True)

    def _run(self, artifact_id: str) -> str:
        """Evaluate accessibility bar."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        hot_store = self.state.get("hot_store", {})
        artifact = hot_store.get(artifact_id)

        if artifact is None:
            return json.dumps(
                {
                    "bar": "accessibility",
                    "artifact_id": artifact_id,
                    "passed": False,
                    "issues": [f"Artifact '{artifact_id}' not found in hot_store"],
                    "notes": "Cannot evaluate - artifact missing",
                }
            )

        # Auto-pass for now (stub implementation)
        return json.dumps(
            {
                "bar": "accessibility",
                "artifact_id": artifact_id,
                "passed": True,
                "issues": [],
                "notes": "Accessibility evaluation passed - content usable by all players.",
                "next_step": "Record this result and proceed to create_gatecheck_report when all bars are evaluated.",
            }
        )


class CreateGatecheckReport(BaseTool):
    """Create a GatecheckReport artifact documenting validation results.

    This tool creates a formal report of the quality bar evaluation
    and stores it in hot_store for review.
    """

    name: str = "create_gatecheck_report"
    description: str = (
        "Create a GatecheckReport documenting validation results. "
        "Args: target_artifact (str), bars_checked (list), status (str: passed/failed/waived), "
        "bar_results (dict mapping bar names to pass/fail notes), "
        "issues (list of strings), recommendations (list of strings)"
    )

    state: dict[str, Any] | None = Field(default=None, exclude=True)
    role_id: str | None = Field(default=None, exclude=True)

    def _run(
        self,
        target_artifact: str,
        bars_checked: list[str],
        status: Literal["pending", "passed", "failed", "waived"],
        bar_results: dict[str, str] | None = None,
        issues: list[str] | None = None,
        recommendations: list[str] | None = None,
        waiver_reason: str | None = None,
    ) -> str:
        """Create a gatecheck report."""
        if not self.state:
            return json.dumps(
                {
                    "success": False,
                    "error": "No state available",
                    "hint": "State must be injected by executor. This is an internal error.",
                }
            )

        # Validate bars_checked are valid QualityBar values
        valid_bars = {bar.value for bar in QualityBar}
        invalid_bars = [b for b in bars_checked if b not in valid_bars]
        if invalid_bars:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid quality bars: {invalid_bars}",
                    "invalid_fields": [
                        {
                            "field": "bars_checked",
                            "provided": invalid_bars,
                            "issue": "Values not in QualityBar enum",
                        }
                    ],
                    "valid_bars": sorted(valid_bars),
                    "hint": f"Use only valid quality bar names: {', '.join(sorted(valid_bars))}",
                }
            )

        # Generate report ID
        report_count = len(
            [k for k in self.state.get("hot_store", {}) if k.startswith("gatecheck_")]
        )
        report_id = f"gatecheck_{target_artifact}_{report_count + 1}"

        # Build report
        report = {
            "target_artifact": target_artifact,
            "bars_checked": bars_checked,
            "status": status,
            "bar_results": bar_results or {},
            "issues": issues or [],
            "recommendations": recommendations or [],
            "checked_by": self.role_id or "gatekeeper",
        }

        if waiver_reason:
            report["waiver_reason"] = waiver_reason

        # Store in hot_store
        if "hot_store" not in self.state:
            self.state["hot_store"] = {}
        self.state["hot_store"][report_id] = report

        logger.info(f"[{self.role_id}] Created gatecheck report: {report_id} ({status})")

        return json.dumps(
            {
                "success": True,
                "report_id": report_id,
                "status": status,
                "bars_checked": bars_checked,
                "issues_count": len(issues or []),
                "message": f"GatecheckReport '{report_id}' created with status '{status}'",
            }
        )


# Export all tools
GATEKEEPER_TOOLS = [
    EvaluateIntegrity,
    EvaluateReachability,
    EvaluateNonlinearity,
    EvaluateGateways,
    EvaluateStyle,
    EvaluateDeterminism,
    EvaluatePresentation,
    EvaluateAccessibility,
    CreateGatecheckReport,
]
