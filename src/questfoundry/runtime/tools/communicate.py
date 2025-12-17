"""
Communicate tool implementation.

Unified communication channel for orchestrator-to-human interaction.
This replaces request_clarification with a more comprehensive tool
that handles all types of human-facing communication.

Message types:
- status: Brief progress update (non-blocking)
- question: Need input from customer (blocks until answered)
- notification: Deliverable ready or milestone reached (non-blocking)
- error: Something failed or needs attention (may block depending on severity)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from questfoundry.runtime.messaging import create_message
from questfoundry.runtime.messaging.types import MessageType
from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


class CommunicationType(str, Enum):
    """Types of communication with the customer."""

    STATUS = "status"
    QUESTION = "question"
    NOTIFICATION = "notification"
    ERROR = "error"


class ErrorSeverity(str, Enum):
    """Severity levels for error communications."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# Map communicate types to message types
_COMMUNICATE_TO_MESSAGE_TYPE = {
    CommunicationType.STATUS: MessageType.PROGRESS_UPDATE,
    CommunicationType.QUESTION: MessageType.CLARIFICATION_REQUEST,
    CommunicationType.NOTIFICATION: MessageType.COMPLETION_SIGNAL,
    CommunicationType.ERROR: MessageType.ESCALATION,
}


@register_tool("communicate")
class CommunicateTool(BaseTool):
    """
    Send a message to the human customer.

    This is the ONLY way orchestrators communicate with humans.
    All output must go through this tool.

    This tool has terminates_turn set to true, meaning calling it ends
    the agent's turn. Orchestrators must end every turn with either
    this tool or delegate.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Create and route a communication message.

        For questions, the response will be provided in the agent's
        next activation. For other types, the message is delivered
        and execution continues.
        """
        comm_type_str = args.get("type")
        message_text = args.get("message")
        context = args.get("context")
        options = args.get("options", [])
        default_option = args.get("default_option")
        artifacts = args.get("artifacts", [])
        severity_str = args.get("severity", "info")

        # Validate required fields
        if not comm_type_str:
            raise ToolValidationError("Communication type is required")
        if not message_text:
            raise ToolValidationError("Message is required")

        # Parse and validate type
        try:
            comm_type = CommunicationType(comm_type_str)
        except ValueError:
            valid_types = [t.value for t in CommunicationType]
            raise ToolValidationError(
                f"Invalid communication type '{comm_type_str}'. Valid types: {valid_types}"
            ) from None

        # Parse severity for errors
        try:
            severity = ErrorSeverity(severity_str)
        except ValueError:
            # For error type, invalid severity is an error; for others, default to INFO
            if comm_type == CommunicationType.ERROR:
                valid_severities = [s.value for s in ErrorSeverity]
                raise ToolValidationError(
                    f"Invalid severity '{severity_str}'. Valid values: {valid_severities}"
                ) from None
            severity = ErrorSeverity.INFO

        # Build payload
        payload: dict[str, Any] = {
            "communication_type": comm_type.value,
            "message": message_text,
        }

        if context:
            payload["context"] = context

        # Type-specific payload fields
        if comm_type == CommunicationType.QUESTION:
            if options:
                payload["options"] = options
            if default_option:
                payload["default_option"] = default_option
            payload["blocking"] = True

        elif comm_type == CommunicationType.NOTIFICATION:
            if artifacts:
                payload["artifacts"] = artifacts
            payload["blocking"] = False

        elif comm_type == CommunicationType.ERROR:
            payload["severity"] = severity.value
            # Errors with 'error' severity may need human intervention
            payload["blocking"] = severity == ErrorSeverity.ERROR

        elif comm_type == CommunicationType.STATUS:
            payload["blocking"] = False

        # Get the appropriate message type
        message_type = _COMMUNICATE_TO_MESSAGE_TYPE[comm_type]

        # Create the message
        message = create_message(
            message_type=message_type,
            from_agent=self._context.agent_id or "unknown",
            to_agent="customer",  # Special agent ref for human
            payload=payload,
        )

        # Route via broker if available
        delivered = False
        if self._context.broker:
            await self._context.broker.send(message)
            delivered = True
            logger.info(
                "Communication %s [%s]: %s sends '%s'",
                message.id,
                comm_type.value,
                self._context.agent_id,
                message_text[:50] + "..." if len(message_text) > 50 else message_text,
            )
        else:
            logger.warning(
                "No broker available - communication %s created but not routed",
                message.id,
            )

        # Build result
        result_data: dict[str, Any] = {
            "message_id": message.id,
            "type": comm_type.value,
            "delivered": delivered,
            "blocking": payload.get("blocking", False),
        }

        # For questions, indicate that response will be provided later
        if comm_type == CommunicationType.QUESTION:
            result_data["status"] = "awaiting_response"
            result_data["message"] = message_text
            if options:
                result_data["options"] = [opt.get("id") for opt in options]
            if default_option:
                result_data["default_option"] = default_option

        # For notifications, include artifact refs
        elif comm_type == CommunicationType.NOTIFICATION:
            result_data["status"] = "delivered" if delivered else "pending"
            if artifacts:
                result_data["artifacts"] = artifacts

        # For errors, include severity
        elif comm_type == CommunicationType.ERROR:
            result_data["status"] = "reported"
            result_data["severity"] = severity.value
            if severity == ErrorSeverity.ERROR:
                result_data["awaiting_decision"] = True

        # For status, simple acknowledgment
        else:
            result_data["status"] = "delivered" if delivered else "pending"

        return ToolResult(
            success=True,
            data=result_data,
        )

    def validate_input(self, args: dict[str, Any]) -> None:
        """Validate input with type-specific rules."""
        # First do base validation
        super().validate_input(args)

        comm_type = args.get("type")
        options = args.get("options", [])
        default_option = args.get("default_option")

        # Validate options structure if provided
        if options:
            if not isinstance(options, list):
                raise ToolValidationError("Options must be an array")

            for i, opt in enumerate(options):
                if not isinstance(opt, dict):
                    raise ToolValidationError(f"Option {i} must be an object")
                if "id" not in opt:
                    raise ToolValidationError(f"Option {i} missing required 'id' field")
                if "description" not in opt:
                    raise ToolValidationError(f"Option {i} missing required 'description' field")

        # If default_option provided, it should match an option id (if options exist)
        if default_option and options:
            # After validation above, we know all options have "id" field
            option_ids = [opt["id"] for opt in options]
            if default_option not in option_ids:
                raise ToolValidationError(
                    f"default_option '{default_option}' not found in options: {option_ids}"
                )

        # Validate severity for errors
        if comm_type == "error":
            severity = args.get("severity", "info")
            valid_severities = [s.value for s in ErrorSeverity]
            if severity not in valid_severities:
                raise ToolValidationError(
                    f"Invalid severity '{severity}'. Valid values: {valid_severities}"
                )
