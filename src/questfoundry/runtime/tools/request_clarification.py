"""
Request Clarification tool implementation.

Allows agents to ask the customer (human) clarifying questions.
This is a stop tool that pauses the workflow and returns control
to the human for input.

The CLI handles displaying the question and capturing the response.
"""

from __future__ import annotations

import logging
from typing import Any

from questfoundry.runtime.messaging import create_message
from questfoundry.runtime.messaging.types import MessageType
from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


@register_tool("request_clarification")
class RequestClarificationTool(BaseTool):
    """
    Ask the customer a clarifying question.

    This pauses the workflow and returns control to the human.
    The response will be provided in the agent's next activation.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Create a clarification request.

        The CLI will:
        1. Display the question to the human
        2. Capture their response
        3. Include it in the next agent activation
        """
        question = args.get("question")
        context = args.get("context")
        options = args.get("options", [])
        default_option = args.get("default_option")

        # Validate: must have question
        if not question:
            raise ToolValidationError("Question is required")

        # Build clarification payload
        payload: dict[str, Any] = {
            "question": question,
        }

        if context:
            payload["context"] = context

        if options:
            payload["options"] = options

        if default_option:
            payload["default_option"] = default_option

        # Create clarification request message
        # Target is "customer" (special agent ref for human)
        message = create_message(
            message_type=MessageType.CLARIFICATION_REQUEST,
            from_agent=self._context.agent_id or "unknown",
            to_agent="customer",
            payload=payload,
        )

        # Route via broker if available
        if self._context.broker:
            await self._context.broker.send(message)
            logger.info(
                "Clarification request %s: %s asks '%s'",
                message.id,
                self._context.agent_id,
                question[:50] + "..." if len(question) > 50 else question,
            )
        else:
            logger.warning(
                "No broker available - clarification request %s created but not routed",
                message.id,
            )

        return ToolResult(
            success=True,
            data={
                "message_id": message.id,
                "status": "pending",
                "question": question,
                "options": options if options else None,
            },
        )
