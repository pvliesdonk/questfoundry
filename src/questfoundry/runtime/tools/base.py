"""
Base tool infrastructure for the QuestFoundry runtime.

Tools are callable units that agents use to interact with the system.
They are defined declaratively in domain/tools/*.json and implemented here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.messaging.broker import AsyncMessageBroker
    from questfoundry.runtime.models import Studio, Tool
    from questfoundry.runtime.storage import LifecycleManager, Project, StoreManager


class ToolError(Exception):
    """Base exception for tool execution errors."""

    pass


class ToolValidationError(ToolError):
    """Tool input validation failed."""

    pass


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    pass


class ToolUnavailableError(ToolError):
    """Tool is not available (missing dependencies, config, etc.)."""

    pass


class CapabilityViolationError(ToolError):
    """Agent attempted to use a tool they don't have capability for."""

    def __init__(self, agent_id: str, tool_id: str):
        self.agent_id = agent_id
        self.tool_id = tool_id
        super().__init__(f"Agent '{agent_id}' lacks capability to use tool '{tool_id}'")


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    data: dict[str, Any]
    error: str | None = None
    execution_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ToolContext:
    """
    Context provided to tool implementations.

    Contains references to runtime services needed by tools.
    """

    studio: Studio
    project: Project | None = None
    agent_id: str | None = None
    session_id: str | None = None

    # Messaging infrastructure (Phase 3)
    broker: AsyncMessageBroker | None = None

    # Storage infrastructure (Phase 4)
    store_manager: StoreManager | None = None
    lifecycle_manager: LifecycleManager | None = None

    # Additional context that may be needed
    domain_path: Path | None = None  # Path to domain directory

    # Interactive mode flag - tools requiring human input should check this
    interactive: bool = True


class BaseTool(ABC):
    """
    Abstract base class for tool implementations.

    Each tool implements:
    - execute(): Core tool logic
    - validate_input(): Input validation (optional override)
    - check_availability(): Check if tool can run (optional override)
    """

    def __init__(self, definition: Tool, context: ToolContext):
        """
        Initialize tool with its definition and runtime context.

        Args:
            definition: Tool definition from domain
            context: Runtime context with services
        """
        self._definition = definition
        self._context = context

    @property
    def id(self) -> str:
        """Tool ID from definition."""
        return self._definition.id

    @property
    def name(self) -> str:
        """Tool name from definition."""
        return self._definition.name

    @property
    def description(self) -> str:
        """Tool description from definition."""
        return self._definition.description

    @property
    def definition(self) -> Tool:
        """Full tool definition."""
        return self._definition

    @property
    def context(self) -> ToolContext:
        """Runtime context."""
        return self._context

    @property
    def timeout_ms(self) -> int:
        """Timeout from definition."""
        return self._definition.timeout_ms

    def validate_input(self, args: dict[str, Any]) -> None:
        """
        Validate input arguments against schema.

        Override for custom validation logic.

        Args:
            args: Input arguments to validate

        Raises:
            ToolValidationError: If validation fails
        """
        if not self._definition.input_schema:
            return

        schema = self._definition.input_schema
        required = schema.required or []

        # Check required fields
        for field_name in required:
            if field_name not in args:
                raise ToolValidationError(f"Missing required field: {field_name}")

        # Check property types (basic validation)
        properties = schema.properties or {}
        for field_name, value in args.items():
            if field_name in properties:
                prop_schema = properties[field_name]
                expected_type = prop_schema.get("type")

                # Basic type checking
                # Note: bool check comes before int because bool is a subclass of int
                if expected_type == "string" and not isinstance(value, str):
                    raise ToolValidationError(f"Field '{field_name}' must be a string")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ToolValidationError(f"Field '{field_name}' must be a boolean")
                elif expected_type == "integer" and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    raise ToolValidationError(f"Field '{field_name}' must be an integer")
                elif expected_type == "array" and not isinstance(value, list):
                    raise ToolValidationError(f"Field '{field_name}' must be an array")
                elif expected_type == "object" and not isinstance(value, dict):
                    raise ToolValidationError(f"Field '{field_name}' must be an object")

    def check_availability(self) -> bool:
        """
        Check if the tool is available to execute.

        Override to check for required services, API keys, etc.

        Returns:
            True if tool can execute, False otherwise
        """
        return True

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            args: Validated input arguments

        Returns:
            ToolResult with execution outcome
        """
        ...

    async def run(self, args: dict[str, Any]) -> ToolResult:
        """
        Full tool execution with validation and timing.

        This is the main entry point for tool execution.

        Args:
            args: Input arguments

        Returns:
            ToolResult with execution outcome
        """
        start_time = time.time()

        try:
            # Check availability
            if not self.check_availability():
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Tool '{self.id}' is not available",
                )

            # Validate input
            self.validate_input(args)

            # Execute
            result = await self.execute(args)

            # Add timing
            result.execution_time_ms = (time.time() - start_time) * 1000

            return result

        except ToolValidationError as e:
            return ToolResult(
                success=False,
                data={},
                error=f"Validation error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except ToolExecutionError as e:
            return ToolResult(
                success=False,
                data={},
                error=f"Execution error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                error=f"Unexpected error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def to_langchain_schema(self) -> dict[str, Any]:
        """
        Convert tool to LangChain-compatible schema for bind_tools().

        Returns:
            Dict with name, description, and parameters
        """
        parameters = {"type": "object", "properties": {}, "required": []}

        if self._definition.input_schema:
            parameters["properties"] = self._definition.input_schema.properties or {}
            parameters["required"] = self._definition.input_schema.required or []

        return {
            "name": self.id,
            "description": self.description,
            "parameters": parameters,
        }


class UnavailableTool(BaseTool):
    """
    Stub tool that returns an unavailable message.

    Used when a tool is defined but not implemented yet.
    """

    def __init__(self, definition: Tool, context: ToolContext, reason: str = "not implemented"):
        super().__init__(definition, context)
        self._reason = reason

    def check_availability(self) -> bool:
        return False

    async def execute(self, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=False,
            data={},
            error=f"Tool '{self.id}' is {self._reason}",
        )
