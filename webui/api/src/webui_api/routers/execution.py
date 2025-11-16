"""Execution endpoints for goal execution and gatechecking

This router provides the core QuestFoundry functionality:
- Execute goals using the orchestrator
- Run gatechecks for quality validation
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..lifecycle import orchestrator_context
from ..user_settings import get_user_provider_config

router = APIRouter(prefix="/projects/{project_id}", tags=["execution"])


class GoalRequest(BaseModel):
    """Request to execute a goal"""

    goal: str = Field(..., description="Goal to execute")
    context: dict[str, Any] | None = Field(
        None, description="Optional context for goal execution"
    )


class GoalResponse(BaseModel):
    """Response from goal execution"""

    status: str = Field(..., description="Execution status")
    result: Any = Field(..., description="Execution result")


class GatecheckRequest(BaseModel):
    """Request to run gatecheck validation"""

    artifacts: list[str] | None = Field(
        None, description="Specific artifact IDs to validate (all if None)"
    )


class GatecheckResponse(BaseModel):
    """Response from gatecheck validation"""

    status: str = Field(..., description="Validation status")
    passed: bool = Field(..., description="Whether validation passed")
    issues: list[dict[str, Any]] = Field(..., description="Validation issues found")


@router.post("/execute", response_model=GoalResponse)
async def execute_goal(
    project_id: str,
    request: Request,
    goal_request: GoalRequest,
) -> GoalResponse:
    """
    Execute a goal using the orchestrator.

    This is the main entry point for QuestFoundry operations. It:
    1. Acquires a distributed lock on the project
    2. Gets the user's provider configuration (BYOK)
    3. Creates an orchestrator with project-scoped storage
    4. Executes the goal
    5. Returns the result

    The lock is automatically released when execution completes.

    Args:
        project_id: Project identifier
        request: FastAPI request (contains user_id from AuthMiddleware)
        goal_request: Goal execution request

    Returns:
        Execution result

    Raises:
        HTTPException: 423 if project is locked by another user
        HTTPException: 500 if execution fails
    """
    user_id = request.state.user_id

    # Get user's provider config (decrypted BYOK keys)
    provider_config = await get_user_provider_config(user_id)

    try:
        # Use orchestrator context (handles locking, storage, lifecycle)
        with orchestrator_context(
            project_id, user_id, provider_config
        ) as orchestrator:
            # Execute goal
            result = orchestrator.execute_goal(
                goal=goal_request.goal,
                context=goal_request.context or {},
            )

            return GoalResponse(
                status="success",
                result=result,
            )
    except Exception as e:
        # Log error and return structured error response
        raise HTTPException(
            status_code=500,
            detail=f"Goal execution failed: {str(e)}",
        ) from e


@router.post("/gatecheck", response_model=GatecheckResponse)
async def run_gatecheck(
    project_id: str,
    request: Request,
    gatecheck_request: GatecheckRequest,
) -> GatecheckResponse:
    """
    Run gatecheck validation on artifacts.

    Validates artifacts against the 8 quality bars defined in the spec.
    This is typically run before merging hot artifacts to cold storage.

    Args:
        project_id: Project identifier
        request: FastAPI request (contains user_id)
        gatecheck_request: Gatecheck request

    Returns:
        Validation results

    Raises:
        HTTPException: 423 if project is locked
        HTTPException: 500 if validation fails
    """
    user_id = request.state.user_id

    # Get user's provider config
    provider_config = await get_user_provider_config(user_id)

    try:
        with orchestrator_context(
            project_id, user_id, provider_config
        ) as _orchestrator:
            # Run gatecheck
            # Note: Actual implementation depends on orchestrator.run_gatecheck method
            # For now, returning placeholder
            issues: list[dict[str, Any]] = []

            # TODO: Implement actual gatecheck logic
            # issues = orchestrator.run_gatecheck(
            #     artifact_ids=gatecheck_request.artifacts
            # )

            passed = len(issues) == 0

            return GatecheckResponse(
                status="success",
                passed=passed,
                issues=issues,
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gatecheck failed: {str(e)}",
        ) from e
