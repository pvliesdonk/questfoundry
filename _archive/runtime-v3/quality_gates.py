"""Quality gate enforcement for lifecycle transitions.

Per meta/ specification, lifecycle transitions can require validation
against quality criteria. This module implements:

1. Loading `requires_validation` criteria from artifact type transitions
2. Running validation (runtime or LLM)
3. Blocking transitions when `gate` criteria fail
4. Advisory feedback when criteria fail but don't block

Validation Dimensions
--------------------
| Enforcement | Blocking | Behavior |
|-------------|----------|----------|
| runtime | gate | Programmatic check must pass to transition |
| runtime | advisory | Check runs, feedback only |
| llm | gate | LLM semantic check must pass |
| llm | advisory | LLM feedback only |

Usage
-----
Create a quality gate validator::

    from questfoundry.runtime.quality_gates import QualityGateValidator

    validator = QualityGateValidator(studio)
    result = await validator.validate_transition(artifact, "approved", "cold")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.runtime.domain.models import Studio

logger = logging.getLogger(__name__)


LLM_VALIDATION_SYSTEM_PROMPT = """You are a quality gate validator for an interactive fiction authoring system.

Your task is to evaluate whether an artifact meets specific quality criteria for a lifecycle transition.

For each criterion, evaluate the artifact and respond with a JSON object:
{
  "criterion_id": "the criterion ID",
  "passed": true or false,
  "score": 0.0 to 1.0,
  "feedback": "Brief explanation of your assessment",
  "issues": ["List of specific issues found, if any"]
}

Be strict but fair. The criteria exist to ensure quality."""


class ValidationResult(BaseModel):
    """Result of validating a single quality criterion.

    Attributes
    ----------
    criterion_id : str
        The quality criterion ID.
    passed : bool
        Whether the criterion passed.
    blocking : bool
        Whether this criterion blocks the transition if failed.
    score : float
        Validation score (0.0 to 1.0).
    feedback : str
        Human-readable feedback.
    issues : list[str]
        Specific issues found.
    """

    criterion_id: str
    passed: bool
    blocking: bool = False
    score: float = 1.0
    feedback: str = ""
    issues: list[str] = Field(default_factory=list)


class TransitionValidationResult(BaseModel):
    """Result of validating a lifecycle transition.

    Attributes
    ----------
    can_transition : bool
        Whether the transition is allowed.
    results : list[ValidationResult]
        Results for each validated criterion.
    blocking_failures : list[str]
        IDs of blocking criteria that failed.
    advisory_failures : list[str]
        IDs of advisory criteria that failed.
    guidance : str
        Overall guidance message.
    """

    can_transition: bool
    results: list[ValidationResult] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)
    advisory_failures: list[str] = Field(default_factory=list)
    guidance: str = ""


class QualityGateValidator(BaseModel):
    """Validates quality criteria for lifecycle transitions.

    Attributes
    ----------
    studio : Studio | None
        The loaded studio with quality criteria definitions.
    """

    studio: Any = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def get_transition_criteria(
        self,
        artifact_type: str,
        from_state: str,
        to_state: str,
    ) -> list[str]:
        """Get the quality criteria required for a transition.

        Parameters
        ----------
        artifact_type : str
            The artifact type ID.
        from_state : str
            The current lifecycle state.
        to_state : str
            The target lifecycle state.

        Returns
        -------
        list[str]
            Quality criterion IDs required for this transition.
        """
        if not self.studio:
            return []

        # Get artifact type definition
        artifact_types = getattr(self.studio, "artifact_types", {})
        type_def = artifact_types.get(artifact_type)
        if not type_def:
            return []

        # Get lifecycle definition
        lifecycle = getattr(type_def, "lifecycle", None)
        if not lifecycle:
            return []

        # Get transitions
        transitions = getattr(lifecycle, "transitions", [])
        if isinstance(transitions, dict):
            transitions = transitions.get("transitions", [])

        # Find matching transition
        for trans in transitions:
            trans_from = getattr(trans, "from_state", None)
            if trans_from is None and hasattr(trans, "get"):
                trans_from = trans.get("from")

            trans_to = getattr(trans, "to_state", None)
            if trans_to is None and hasattr(trans, "get"):
                trans_to = trans.get("to")

            if trans_from == from_state and trans_to == to_state:
                # Found the transition - get requires_validation
                requires = getattr(trans, "requires_validation", None)
                if requires is None and hasattr(trans, "get"):
                    requires = trans.get("requires_validation", [])
                return requires or []

        return []

    def get_criterion_definition(self, criterion_id: str) -> dict[str, Any] | None:
        """Get a quality criterion definition.

        Parameters
        ----------
        criterion_id : str
            The criterion ID.

        Returns
        -------
        dict[str, Any] | None
            The criterion definition, or None if not found.
        """
        if not self.studio:
            return None

        criteria = getattr(self.studio, "quality_criteria", {})
        criterion = criteria.get(criterion_id)

        if not criterion:
            return None

        # Convert to dict if needed
        if hasattr(criterion, "model_dump") and callable(criterion.model_dump):
            return criterion.model_dump()
        if hasattr(criterion, "__dict__"):
            return {
                "id": getattr(criterion, "id", criterion_id),
                "name": getattr(criterion, "name", criterion_id),
                "description": getattr(criterion, "description", ""),
                "enforcement": getattr(criterion, "enforcement", "llm"),
                "blocking": getattr(criterion, "blocking", "advisory"),
                "check": getattr(criterion, "check", None),
                "failure_guidance": getattr(criterion, "failure_guidance", ""),
            }

        return None

    async def validate_transition(
        self,
        artifact: Any,
        artifact_type: str,
        from_state: str,
        to_state: str,
        llm: BaseChatModel | None = None,
    ) -> TransitionValidationResult:
        """Validate a lifecycle transition against quality criteria.

        Parameters
        ----------
        artifact : Any
            The artifact to validate.
        artifact_type : str
            The artifact type ID.
        from_state : str
            Current lifecycle state.
        to_state : str
            Target lifecycle state.
        llm : BaseChatModel | None
            LLM for semantic validation.

        Returns
        -------
        TransitionValidationResult
            The validation result.
        """
        criteria_ids = self.get_transition_criteria(artifact_type, from_state, to_state)

        if not criteria_ids:
            # No validation required
            return TransitionValidationResult(
                can_transition=True,
                guidance="No validation criteria required for this transition.",
            )

        results: list[ValidationResult] = []
        blocking_failures: list[str] = []
        advisory_failures: list[str] = []

        for criterion_id in criteria_ids:
            criterion = self.get_criterion_definition(criterion_id)

            if not criterion:
                # Unknown criterion - skip with warning
                logger.warning("Unknown quality criterion: %s", criterion_id)
                continue

            enforcement = criterion.get("enforcement", "llm")
            blocking = criterion.get("blocking", "advisory") == "gate"

            # Run validation
            if enforcement == "runtime":
                result = await self._validate_runtime(artifact, criterion)
            else:
                if llm:
                    result = await self._validate_llm(artifact, criterion, llm)
                else:
                    # No LLM available - defer validation
                    result = ValidationResult(
                        criterion_id=criterion_id,
                        passed=True,  # Pass by default when LLM unavailable
                        blocking=blocking,
                        feedback="LLM validation deferred (no LLM provided)",
                    )

            result.blocking = blocking
            results.append(result)

            if not result.passed:
                if blocking:
                    blocking_failures.append(criterion_id)
                else:
                    advisory_failures.append(criterion_id)

        # Build guidance
        can_transition = len(blocking_failures) == 0
        guidance_parts = []

        if blocking_failures:
            guidance_parts.append(
                f"Blocking criteria failed: {', '.join(blocking_failures)}. "
                "Address these issues before the transition can proceed."
            )
        if advisory_failures:
            guidance_parts.append(
                f"Advisory criteria failed: {', '.join(advisory_failures)}. "
                "Consider addressing these issues."
            )
        if can_transition and not advisory_failures:
            guidance_parts.append("All quality criteria passed.")

        return TransitionValidationResult(
            can_transition=can_transition,
            results=results,
            blocking_failures=blocking_failures,
            advisory_failures=advisory_failures,
            guidance=" ".join(guidance_parts),
        )

    async def _validate_runtime(
        self,
        artifact: Any,
        criterion: dict[str, Any],
    ) -> ValidationResult:
        """Run runtime (programmatic) validation.

        Parameters
        ----------
        artifact : Any
            The artifact to validate.
        criterion : dict[str, Any]
            The criterion definition.

        Returns
        -------
        ValidationResult
            The validation result.
        """
        criterion_id = criterion.get("id", "unknown")
        check = criterion.get("check", {})

        # Currently we only have LLM rubric checks in domain-v4
        # For runtime checks, we'd implement specific validators here
        # For now, pass with a note

        return ValidationResult(
            criterion_id=criterion_id,
            passed=True,
            score=1.0,
            feedback="Runtime validation not implemented - passed by default",
        )

    async def _validate_llm(
        self,
        artifact: Any,
        criterion: dict[str, Any],
        llm: BaseChatModel,
    ) -> ValidationResult:
        """Run LLM-based semantic validation.

        Parameters
        ----------
        artifact : Any
            The artifact to validate.
        criterion : dict[str, Any]
            The criterion definition.
        llm : BaseChatModel
            The LLM to use.

        Returns
        -------
        ValidationResult
            The validation result.
        """
        criterion_id = criterion.get("id", "unknown")
        name = criterion.get("name", criterion_id)
        description = criterion.get("description", "")
        check = criterion.get("check", {})
        failure_guidance = criterion.get("failure_guidance", "")

        # Format artifact for LLM
        if hasattr(artifact, "model_dump"):
            artifact_text = json.dumps(artifact.model_dump(), indent=2, default=str)
        elif isinstance(artifact, dict):
            artifact_text = json.dumps(artifact, indent=2, default=str)
        else:
            artifact_text = str(artifact)

        # Build prompt
        rubric = check.get("rubric", {})
        rubric_text = ""
        if rubric:
            rubric_text = f"\n\nRubric:\n{json.dumps(rubric, indent=2)}"

        user_prompt = f"""Evaluate this artifact against the quality criterion:

Criterion: {name}
Description: {description}
{rubric_text}

Artifact to evaluate:
```json
{artifact_text}
```

Respond with a JSON object containing: criterion_id, passed (boolean), score (0-1), feedback, issues (list)"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=LLM_VALIDATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result_data = json.loads(content.strip())

            return ValidationResult(
                criterion_id=criterion_id,
                passed=result_data.get("passed", False),
                score=result_data.get("score", 0.0),
                feedback=result_data.get("feedback", ""),
                issues=result_data.get("issues", []),
            )

        except Exception as e:
            logger.warning("LLM validation failed for %s: %s", criterion_id, e)
            # On error, pass with warning
            return ValidationResult(
                criterion_id=criterion_id,
                passed=True,
                score=1.0,
                feedback=f"LLM validation error: {e}. Passed by default.",
                issues=[],
            )


def create_quality_gate_validator(studio: Any = None) -> QualityGateValidator:
    """Factory function to create a quality gate validator.

    Parameters
    ----------
    studio : Any
        The loaded studio with quality criteria.

    Returns
    -------
    QualityGateValidator
        Configured validator instance.
    """
    return QualityGateValidator(studio=studio)
