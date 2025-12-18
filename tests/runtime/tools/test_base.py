"""Tests for tool base classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.tools.base import (
    BaseTool,
    ToolContext,
    ToolExecutionError,
    ToolResult,
    ToolValidationError,
    UnavailableTool,
)


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._execute_called = False
        self._execute_args: dict[str, Any] = {}
        self._return_value: ToolResult | None = None
        self._raise_error: Exception | None = None

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        self._execute_called = True
        self._execute_args = args

        if self._raise_error:
            raise self._raise_error

        if self._return_value:
            return self._return_value

        return ToolResult(success=True, data={"echo": args})


def make_mock_tool_definition():
    """Create a mock tool definition."""
    mock_def = MagicMock()
    mock_def.id = "test_tool"
    mock_def.name = "Test Tool"
    mock_def.description = "A test tool"
    mock_def.timeout_ms = 30000

    # Input schema
    mock_schema = MagicMock()
    mock_schema.properties = {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "active": {"type": "boolean"},
    }
    mock_schema.required = ["name"]
    mock_def.input_schema = mock_schema

    return mock_def


def make_mock_context():
    """Create a mock tool context."""
    mock_studio = MagicMock()
    return ToolContext(studio=mock_studio)


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, data={}, error="Something failed")
        assert result.success is False
        assert result.error == "Something failed"

    def test_to_dict(self):
        result = ToolResult(
            success=True,
            data={"key": "value"},
            execution_time_ms=123.45,
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["data"] == {"key": "value"}
        assert d["error"] is None
        assert d["execution_time_ms"] == 123.45
        assert d["fatal"] is False

    def test_to_dict_includes_fatal_flag(self):
        result = ToolResult(success=False, data={}, fatal=True)
        assert result.to_dict()["fatal"] is True


class TestBaseTool:
    """Tests for BaseTool."""

    def test_properties(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        assert tool.id == "test_tool"
        assert tool.name == "Test Tool"
        assert tool.description == "A test tool"
        assert tool.timeout_ms == 30000

    def test_validate_input_required_missing(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        with pytest.raises(ToolValidationError, match="Missing required field: name"):
            tool.validate_input({})

    def test_validate_input_required_present(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        # Should not raise
        tool.validate_input({"name": "test"})

    def test_validate_input_wrong_type_string(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        with pytest.raises(ToolValidationError, match="must be a string"):
            tool.validate_input({"name": 123})

    def test_validate_input_wrong_type_integer(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        with pytest.raises(ToolValidationError, match="must be an integer"):
            tool.validate_input({"name": "test", "count": "not_int"})

    def test_validate_input_wrong_type_boolean(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        with pytest.raises(ToolValidationError, match="must be a boolean"):
            tool.validate_input({"name": "test", "active": "yes"})

    def test_check_availability_default(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        assert tool.check_availability() is True

    @pytest.mark.asyncio
    async def test_execute(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        result = await tool.execute({"name": "test"})

        assert result.success is True
        assert result.data == {"echo": {"name": "test"}}

    @pytest.mark.asyncio
    async def test_run_calls_execute(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        result = await tool.run({"name": "test"})

        assert tool._execute_called is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_adds_timing(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        result = await tool.run({"name": "test"})

        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_run_validation_error(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        result = await tool.run({})  # Missing required 'name'

        assert result.success is False
        assert "Validation error" in result.error

    @pytest.mark.asyncio
    async def test_run_execution_error(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)
        tool._raise_error = ToolExecutionError("Boom!")

        result = await tool.run({"name": "test"})

        assert result.success is False
        assert "Execution error" in result.error
        assert "Boom!" in result.error

    def test_to_langchain_schema(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = MockTool(definition, context)

        schema = tool.to_langchain_schema()

        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "parameters" in schema
        assert schema["parameters"]["required"] == ["name"]


class TestUnavailableTool:
    """Tests for UnavailableTool stub."""

    def test_check_availability(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = UnavailableTool(definition, context, reason="not implemented")

        assert tool.check_availability() is False

    @pytest.mark.asyncio
    async def test_execute(self):
        definition = make_mock_tool_definition()
        context = make_mock_context()
        tool = UnavailableTool(definition, context, reason="coming soon")

        result = await tool.execute({})

        assert result.success is False
        assert "coming soon" in result.error
