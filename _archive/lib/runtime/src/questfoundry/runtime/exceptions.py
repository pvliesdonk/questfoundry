"""
Custom exception hierarchy for QuestFoundry runtime.

All runtime exceptions inherit from QuestFoundryError for easy catching.
"""

from __future__ import annotations


class QuestFoundryError(Exception):
    """
    Base exception for all QuestFoundry runtime errors.

    Supports optional suggestions for resolution.
    """

    def __init__(self, message: str, suggestions: list[str] | None = None):
        """
        Initialize exception with message and optional suggestions.

        Args:
            message: Error description
            suggestions: List of suggested resolutions
        """
        super().__init__(message)
        self.message = message
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        """Format error with suggestions if available."""
        parts = [self.message]
        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")
        return "".join(parts)


class SchemaValidationError(QuestFoundryError):
    """
    Raised when schema validation fails.

    Examples:
    - Invalid role profile schema
    - Invalid loop definition schema
    - Invalid quality gate schema
    """

    def __init__(
        self,
        message: str,
        schema_path: str | None = None,
        validation_errors: list[str] | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize schema validation error.

        Args:
            message: Error description
            schema_path: Path to schema that failed validation
            validation_errors: List of specific validation errors
            suggestions: List of suggested resolutions
        """
        super().__init__(message, suggestions)
        self.schema_path = schema_path
        self.validation_errors = validation_errors or []

    def __str__(self) -> str:
        """Format error with schema path and validation errors."""
        parts = [self.message]

        if self.schema_path:
            parts.append(f"\nSchema: {self.schema_path}")

        if self.validation_errors:
            parts.append("\n\nValidation errors:")
            for error in self.validation_errors:
                parts.append(f"  • {error}")

        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "".join(parts)


class ResourceLoadError(QuestFoundryError):
    """
    Raised when resource loading fails.

    Examples:
    - Role profile not found
    - Loop definition not found
    - Template file not found
    - Schema file not found
    """

    def __init__(
        self,
        message: str,
        resource_path: str | None = None,
        available_resources: list[str] | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize resource load error.

        Args:
            message: Error description
            resource_path: Path to resource that failed to load
            available_resources: List of available resources (for discovery)
            suggestions: List of suggested resolutions
        """
        super().__init__(message, suggestions)
        self.resource_path = resource_path
        self.available_resources = available_resources or []

    def __str__(self) -> str:
        """Format error with resource path and available alternatives."""
        parts = [self.message]

        if self.resource_path:
            parts.append(f"\nRequested: {self.resource_path}")

        if self.available_resources:
            parts.append("\n\nAvailable resources:")
            for resource in self.available_resources[:10]:  # Limit to first 10
                parts.append(f"  • {resource}")
            if len(self.available_resources) > 10:
                parts.append(f"  ... and {len(self.available_resources) - 10} more")

        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "".join(parts)


class GraphBuildError(QuestFoundryError):
    """
    Raised when graph construction fails.

    Examples:
    - Invalid node configuration
    - Invalid edge configuration
    - Circular dependencies detected
    - Missing required nodes
    """

    def __init__(
        self,
        message: str,
        loop_id: str | None = None,
        node_id: str | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize graph build error.

        Args:
            message: Error description
            loop_id: Loop ID where error occurred
            node_id: Node ID where error occurred
            suggestions: List of suggested resolutions
        """
        super().__init__(message, suggestions)
        self.loop_id = loop_id
        self.node_id = node_id

    def __str__(self) -> str:
        """Format error with loop and node context."""
        parts = [self.message]

        if self.loop_id:
            parts.append(f"\nLoop: {self.loop_id}")

        if self.node_id:
            parts.append(f"\nNode: {self.node_id}")

        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "".join(parts)


class StateError(QuestFoundryError):
    """
    Raised when state management fails.

    Examples:
    - Invalid state transition
    - Missing required state fields
    - State mutation errors
    - Concurrent update conflicts
    """

    def __init__(
        self,
        message: str,
        tu_id: str | None = None,
        current_state: str | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize state error.

        Args:
            message: Error description
            tu_id: Trace Unit ID where error occurred
            current_state: Current state value (e.g., lifecycle stage)
            suggestions: List of suggested resolutions
        """
        super().__init__(message, suggestions)
        self.tu_id = tu_id
        self.current_state = current_state

    def __str__(self) -> str:
        """Format error with TU and state context."""
        parts = [self.message]

        if self.tu_id:
            parts.append(f"\nTU ID: {self.tu_id}")

        if self.current_state:
            parts.append(f"\nCurrent state: {self.current_state}")

        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "".join(parts)


class ProviderError(QuestFoundryError):
    """
    Raised when LLM provider operations fail.

    Examples:
    - Provider not configured
    - Invalid API key
    - Rate limit exceeded
    - Model not available
    """

    def __init__(
        self,
        message: str,
        provider_name: str | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize provider error.

        Args:
            message: Error description
            provider_name: Name of the provider (e.g., "anthropic", "openai")
            suggestions: List of suggested resolutions
        """
        super().__init__(message, suggestions)
        self.provider_name = provider_name

    def __str__(self) -> str:
        """Format error with provider context."""
        parts = [self.message]

        if self.provider_name:
            parts.append(f"\nProvider: {self.provider_name}")

        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "".join(parts)


class TemplateRenderError(QuestFoundryError):
    """
    Raised when template rendering fails.

    Examples:
    - Template file not found
    - Missing template variables
    - Template syntax errors
    - Undefined variables in template
    """

    def __init__(
        self,
        message: str,
        template_name: str | None = None,
        missing_vars: list[str] | None = None,
        suggestions: list[str] | None = None,
    ):
        """
        Initialize template render error.

        Args:
            message: Error description
            template_name: Name of template that failed to render
            missing_vars: List of missing template variables
            suggestions: List of suggested resolutions
        """
        super().__init__(message, suggestions)
        self.template_name = template_name
        self.missing_vars = missing_vars or []

    def __str__(self) -> str:
        """Format error with template and missing variables context."""
        parts = [self.message]

        if self.template_name:
            parts.append(f"\nTemplate: {self.template_name}")

        if self.missing_vars:
            parts.append("\n\nMissing variables:")
            for var in self.missing_vars:
                parts.append(f"  • {var}")

        if self.suggestions:
            parts.append("\n\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  • {suggestion}")

        return "".join(parts)
