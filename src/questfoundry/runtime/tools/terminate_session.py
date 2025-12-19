"""
Terminate session tool implementation.

This is the ONLY way for orchestrators to explicitly end a session.
Other tools like communicate and delegate yield control but do NOT end the session.

Use when:
- All requested work is complete
- User explicitly requests session end
- Unrecoverable error prevents further progress
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


class TerminationReason(str, Enum):
    """Reasons for terminating a session."""

    COMPLETE = "complete"
    USER_REQUESTED = "user_requested"
    ERROR = "error"
    BLOCKED = "blocked"


@register_tool("terminate_session")
class TerminateSessionTool(BaseTool):
    """
    End the current session explicitly.

    This tool signals that the orchestrator has finished all work and the
    session should end. Unlike communicate or delegate which yield control
    temporarily, this tool ends the session permanently.

    Only orchestrators should have this capability.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Process session termination request.

        The actual session termination is handled by the runtime when it sees
        that a tool with terminates_session=True was called. This tool just
        returns the termination metadata for logging and audit purposes.
        """
        reason_str = args.get("reason")
        summary = args.get("summary")
        artifacts = args.get("artifacts_produced", [])

        # Validate required fields
        if not reason_str:
            raise ToolValidationError("Termination reason is required")
        if not summary:
            raise ToolValidationError("Summary is required")

        # Parse and validate reason
        try:
            reason = TerminationReason(reason_str)
        except ValueError:
            valid_reasons = [r.value for r in TerminationReason]
            raise ToolValidationError(
                f"Invalid reason '{reason_str}'. Valid reasons: {valid_reasons}"
            ) from None

        logger.info(
            "Session termination requested: reason=%s, artifacts=%d, summary=%s",
            reason.value,
            len(artifacts),
            summary[:100] + "..." if len(summary) > 100 else summary,
        )

        return ToolResult(
            success=True,
            data={
                "status": "terminated",
                "reason": reason.value,
                "summary": summary,
                "artifacts_produced": artifacts,
            },
        )

    def validate_input(self, args: dict[str, Any]) -> None:
        """Validate input."""
        super().validate_input(args)

        reason = args.get("reason")
        if reason:
            valid_reasons = [r.value for r in TerminationReason]
            if reason not in valid_reasons:
                raise ToolValidationError(
                    f"Invalid reason '{reason}'. Valid reasons: {valid_reasons}"
                )

        artifacts = args.get("artifacts_produced")
        if artifacts is not None and not isinstance(artifacts, list):
            raise ToolValidationError("artifacts_produced must be an array")
