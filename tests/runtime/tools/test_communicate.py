"""
Tests for the communicate tool.

Tests the unified communication channel for orchestrator-to-human interaction.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.messaging.types import MessageType
from questfoundry.runtime.tools.base import ToolContext, ToolValidationError
from questfoundry.runtime.tools.communicate import (
    CommunicateTool,
    CommunicationType,
    ErrorSeverity,
)


@pytest.fixture
def mock_studio() -> MagicMock:
    """Create a mock studio."""
    studio = MagicMock()
    studio.id = "test_studio"
    return studio


@pytest.fixture
def mock_broker() -> AsyncMock:
    """Create a mock message broker."""
    broker = AsyncMock()
    broker.send = AsyncMock()
    return broker


@pytest.fixture
def mock_tool_definition() -> MagicMock:
    """Create a mock tool definition for communicate."""
    definition = MagicMock()
    definition.id = "communicate"
    definition.name = "Communicate with Customer"
    definition.description = "Send a message to the human customer."
    definition.timeout_ms = 30000
    definition.input_schema = MagicMock()
    definition.input_schema.required = ["type", "message"]
    definition.input_schema.properties = {
        "type": {"type": "string", "enum": ["status", "question", "notification", "error"]},
        "message": {"type": "string"},
        "context": {"type": "string"},
        "options": {"type": "array"},
        "default_option": {"type": "string"},
        "artifacts": {"type": "array"},
        "severity": {"type": "string", "enum": ["info", "warning", "error"], "default": "info"},
    }
    return definition


@pytest.fixture
def tool_context(mock_studio: MagicMock, mock_broker: AsyncMock) -> ToolContext:
    """Create a tool context with broker."""
    return ToolContext(
        studio=mock_studio,
        agent_id="showrunner",
        session_id="test_session",
        broker=mock_broker,
    )


@pytest.fixture
def tool_context_no_broker(mock_studio: MagicMock) -> ToolContext:
    """Create a tool context without broker."""
    return ToolContext(
        studio=mock_studio,
        agent_id="showrunner",
        session_id="test_session",
        broker=None,
    )


@pytest.fixture
def communicate_tool(
    mock_tool_definition: MagicMock,
    tool_context: ToolContext,
) -> CommunicateTool:
    """Create a communicate tool instance."""
    return CommunicateTool(mock_tool_definition, tool_context)


@pytest.fixture
def communicate_tool_no_broker(
    mock_tool_definition: MagicMock,
    tool_context_no_broker: ToolContext,
) -> CommunicateTool:
    """Create a communicate tool instance without broker."""
    return CommunicateTool(mock_tool_definition, tool_context_no_broker)


class TestCommunicateToolValidation:
    """Tests for input validation."""

    async def test_missing_type_raises_error(self, communicate_tool: CommunicateTool) -> None:
        """Missing type should raise validation error."""
        with pytest.raises(ToolValidationError, match="type is required"):
            await communicate_tool.execute({"message": "Hello"})

    async def test_missing_message_raises_error(self, communicate_tool: CommunicateTool) -> None:
        """Missing message should raise validation error."""
        with pytest.raises(ToolValidationError, match="Message is required"):
            await communicate_tool.execute({"type": "status"})

    async def test_invalid_type_raises_error(self, communicate_tool: CommunicateTool) -> None:
        """Invalid communication type should raise validation error."""
        with pytest.raises(ToolValidationError, match="Invalid communication type"):
            await communicate_tool.execute({"type": "invalid", "message": "Hello"})

    def test_options_must_be_array(self, communicate_tool: CommunicateTool) -> None:
        """Options must be an array if provided."""
        with pytest.raises(ToolValidationError, match="must be an array"):
            communicate_tool.validate_input(
                {
                    "type": "question",
                    "message": "Choose one",
                    "options": "not an array",
                }
            )

    def test_option_must_have_id(self, communicate_tool: CommunicateTool) -> None:
        """Each option must have an id field."""
        with pytest.raises(ToolValidationError, match="missing required 'id' field"):
            communicate_tool.validate_input(
                {
                    "type": "question",
                    "message": "Choose one",
                    "options": [{"description": "Option without id"}],
                }
            )

    def test_option_must_have_description(self, communicate_tool: CommunicateTool) -> None:
        """Each option must have a description field."""
        with pytest.raises(ToolValidationError, match="missing required 'description' field"):
            communicate_tool.validate_input(
                {
                    "type": "question",
                    "message": "Choose one",
                    "options": [{"id": "opt1"}],
                }
            )

    def test_default_option_must_match(self, communicate_tool: CommunicateTool) -> None:
        """default_option must match an option id."""
        with pytest.raises(ToolValidationError, match="not found in options"):
            communicate_tool.validate_input(
                {
                    "type": "question",
                    "message": "Choose one",
                    "options": [
                        {"id": "a", "description": "Option A"},
                        {"id": "b", "description": "Option B"},
                    ],
                    "default_option": "nonexistent",
                }
            )

    def test_valid_options_pass(self, communicate_tool: CommunicateTool) -> None:
        """Valid options should pass validation."""
        # Should not raise
        communicate_tool.validate_input(
            {
                "type": "question",
                "message": "Choose one",
                "options": [
                    {"id": "a", "description": "Option A"},
                    {"id": "b", "description": "Option B"},
                ],
                "default_option": "a",
            }
        )

    def test_invalid_severity_raises_error(self, communicate_tool: CommunicateTool) -> None:
        """Invalid severity for errors should raise validation error."""
        with pytest.raises(ToolValidationError, match="Invalid severity"):
            communicate_tool.validate_input(
                {
                    "type": "error",
                    "message": "Something broke",
                    "severity": "critical",  # not valid
                }
            )


class TestCommunicateToolStatusMessage:
    """Tests for status message type."""

    async def test_status_message_success(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Status message should be delivered and return success."""
        result = await communicate_tool.execute(
            {
                "type": "status",
                "message": "Starting story generation...",
            }
        )

        assert result.success is True
        assert result.data["type"] == "status"
        assert result.data["delivered"] is True
        assert result.data["blocking"] is False
        assert result.data["status"] == "delivered"

        # Verify broker was called
        mock_broker.send.assert_called_once()
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.payload["communication_type"] == "status"
        assert sent_message.payload["message"] == "Starting story generation..."

    async def test_status_with_context(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Status message can include context."""
        result = await communicate_tool.execute(
            {
                "type": "status",
                "message": "Delegating to Plotwright",
                "context": "Story structure needs to be planned first",
            }
        )

        assert result.success is True
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.payload["context"] == "Story structure needs to be planned first"


class TestCommunicateToolQuestionMessage:
    """Tests for question message type."""

    async def test_question_basic(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Question message should be blocking and await response."""
        result = await communicate_tool.execute(
            {
                "type": "question",
                "message": "What tone do you prefer?",
            }
        )

        assert result.success is True
        assert result.data["type"] == "question"
        assert result.data["blocking"] is True
        assert result.data["status"] == "awaiting_response"
        assert result.data["message"] == "What tone do you prefer?"

    async def test_question_with_options(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Question with options should include option ids in result."""
        result = await communicate_tool.execute(
            {
                "type": "question",
                "message": "What tone do you prefer?",
                "options": [
                    {"id": "dark", "description": "Dark and serious"},
                    {"id": "light", "description": "Light and fun"},
                ],
                "default_option": "light",
            }
        )

        assert result.success is True
        assert result.data["options"] == ["dark", "light"]
        assert result.data["default_option"] == "light"

        # Check broker message
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.type == MessageType.CLARIFICATION_REQUEST
        assert len(sent_message.payload["options"]) == 2

    async def test_question_with_context(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Question should include context explaining why input is needed."""
        result = await communicate_tool.execute(
            {
                "type": "question",
                "message": "How many chapters?",
                "context": "This determines the story's scope and pacing",
            }
        )

        assert result.success is True
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.payload["context"] == "This determines the story's scope and pacing"


class TestCommunicateToolNotificationMessage:
    """Tests for notification message type."""

    async def test_notification_basic(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Notification should be non-blocking."""
        result = await communicate_tool.execute(
            {
                "type": "notification",
                "message": "Chapter 1 draft is ready.",
            }
        )

        assert result.success is True
        assert result.data["type"] == "notification"
        assert result.data["blocking"] is False
        assert result.data["status"] == "delivered"

    async def test_notification_with_artifacts(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Notification can include artifact references."""
        result = await communicate_tool.execute(
            {
                "type": "notification",
                "message": "Chapter 1 draft is ready for review.",
                "artifacts": [
                    "workspace:section:chapter_1_v1",
                    "workspace:section:chapter_1_notes",
                ],
            }
        )

        assert result.success is True
        assert result.data["artifacts"] == [
            "workspace:section:chapter_1_v1",
            "workspace:section:chapter_1_notes",
        ]

        # Check broker message
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.type == MessageType.COMPLETION_SIGNAL
        assert sent_message.payload["artifacts"] == [
            "workspace:section:chapter_1_v1",
            "workspace:section:chapter_1_notes",
        ]


class TestCommunicateToolErrorMessage:
    """Tests for error message type."""

    async def test_error_info_severity(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Error with info severity should be non-blocking."""
        result = await communicate_tool.execute(
            {
                "type": "error",
                "message": "Minor issue encountered.",
                "severity": "info",
            }
        )

        assert result.success is True
        assert result.data["type"] == "error"
        assert result.data["severity"] == "info"
        assert result.data["blocking"] is False
        assert "awaiting_decision" not in result.data

    async def test_error_warning_severity(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Error with warning severity should be non-blocking."""
        result = await communicate_tool.execute(
            {
                "type": "error",
                "message": "Could not reach external service.",
                "context": "Web search failed, using cached data.",
                "severity": "warning",
            }
        )

        assert result.success is True
        assert result.data["severity"] == "warning"
        assert result.data["blocking"] is False

    async def test_error_error_severity(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Error with error severity should be blocking."""
        result = await communicate_tool.execute(
            {
                "type": "error",
                "message": "Critical failure in story generation.",
                "context": "Cannot proceed without user decision.",
                "severity": "error",
            }
        )

        assert result.success is True
        assert result.data["severity"] == "error"
        assert result.data["blocking"] is True
        assert result.data["awaiting_decision"] is True

    async def test_error_default_severity(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Error without severity should default to info."""
        result = await communicate_tool.execute(
            {
                "type": "error",
                "message": "Something happened.",
            }
        )

        assert result.success is True
        assert result.data["severity"] == "info"
        assert result.data["blocking"] is False

    async def test_error_invalid_severity_raises(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Error with invalid severity should raise validation error."""
        with pytest.raises(ToolValidationError, match="Invalid severity"):
            await communicate_tool.execute(
                {
                    "type": "error",
                    "message": "Something broke.",
                    "severity": "critical",  # not a valid severity
                }
            )

    async def test_status_invalid_severity_defaults_to_info(
        self,
        communicate_tool: CommunicateTool,
    ) -> None:
        """Non-error types with invalid severity should default to info."""
        # For non-error types, invalid severity is silently ignored
        result = await communicate_tool.execute(
            {
                "type": "status",
                "message": "Progress update",
                "severity": "critical",  # invalid but ignored for status
            }
        )

        assert result.success is True
        # Status doesn't use severity in output, so just verify success


class TestCommunicateToolMessageRouting:
    """Tests for message routing behavior."""

    async def test_message_routed_to_customer(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """All messages should be routed to 'customer' agent."""
        await communicate_tool.execute(
            {
                "type": "status",
                "message": "Test message",
            }
        )

        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.to_agent == "customer"

    async def test_message_from_agent(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Messages should be from the current agent."""
        await communicate_tool.execute(
            {
                "type": "status",
                "message": "Test message",
            }
        )

        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.from_agent == "showrunner"

    async def test_no_broker_still_succeeds(
        self,
        communicate_tool_no_broker: CommunicateTool,
    ) -> None:
        """Tool should succeed even without broker (just not delivered)."""
        result = await communicate_tool_no_broker.execute(
            {
                "type": "status",
                "message": "Test message",
            }
        )

        assert result.success is True
        assert result.data["delivered"] is False
        assert result.data["status"] == "pending"

    async def test_question_no_broker(
        self,
        communicate_tool_no_broker: CommunicateTool,
    ) -> None:
        """Question without broker should still return awaiting_response."""
        result = await communicate_tool_no_broker.execute(
            {
                "type": "question",
                "message": "What do you think?",
            }
        )

        assert result.success is True
        assert result.data["delivered"] is False
        assert result.data["status"] == "awaiting_response"


class TestCommunicateToolMessageTypes:
    """Tests for message type mapping."""

    async def test_status_uses_progress_update_type(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Status should map to PROGRESS_UPDATE message type."""
        await communicate_tool.execute({"type": "status", "message": "Test"})
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.type == MessageType.PROGRESS_UPDATE

    async def test_question_uses_clarification_request_type(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Question should map to CLARIFICATION_REQUEST message type."""
        await communicate_tool.execute({"type": "question", "message": "Test?"})
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.type == MessageType.CLARIFICATION_REQUEST

    async def test_notification_uses_completion_signal_type(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Notification should map to COMPLETION_SIGNAL message type."""
        await communicate_tool.execute({"type": "notification", "message": "Done"})
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.type == MessageType.COMPLETION_SIGNAL

    async def test_error_uses_escalation_type(
        self,
        communicate_tool: CommunicateTool,
        mock_broker: AsyncMock,
    ) -> None:
        """Error should map to ESCALATION message type."""
        await communicate_tool.execute({"type": "error", "message": "Failed"})
        sent_message = mock_broker.send.call_args[0][0]
        assert sent_message.type == MessageType.ESCALATION


class TestCommunicationTypeEnum:
    """Tests for the CommunicationType enum."""

    def test_all_types_defined(self) -> None:
        """All expected communication types should be defined."""
        assert CommunicationType.STATUS.value == "status"
        assert CommunicationType.QUESTION.value == "question"
        assert CommunicationType.NOTIFICATION.value == "notification"
        assert CommunicationType.ERROR.value == "error"

    def test_parse_from_string(self) -> None:
        """Should parse from string value."""
        assert CommunicationType("status") == CommunicationType.STATUS
        assert CommunicationType("question") == CommunicationType.QUESTION


class TestErrorSeverityEnum:
    """Tests for the ErrorSeverity enum."""

    def test_all_severities_defined(self) -> None:
        """All expected severity levels should be defined."""
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
