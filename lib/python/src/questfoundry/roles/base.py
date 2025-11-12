"""Base classes for QuestFoundry roles."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models.artifact import Artifact
from ..providers.audio import AudioProvider
from ..providers.base import ImageProvider, TextProvider
from .human_callback import batch_mode_callback

logger = logging.getLogger(__name__)

# Avoid circular imports
if TYPE_CHECKING:
    from .human_callback import HumanCallback
    from .session import RoleSession

# Maximum length for artifact value strings in formatted output
MAX_ARTIFACT_VALUE_LENGTH = 500

# Maximum number of artifacts to include in formatted output
MAX_ARTIFACTS_IN_CONTEXT = 50

# Maximum total size of formatted artifacts context (characters)
MAX_FORMATTED_CONTEXT_SIZE = 50000


@dataclass
class RoleContext:
    """
    Context provided to a role for task execution.

    This contains all information needed for a role to perform its work,
    including artifacts, project metadata, and configuration.
    """

    task: str
    """The specific task to execute (e.g., 'generate_hooks', 'validate_tu')"""

    artifacts: list[Artifact] = field(default_factory=list)
    """Input artifacts available to the role"""

    project_metadata: dict[str, Any] = field(default_factory=dict)
    """Project-level configuration and metadata"""

    workspace_path: Path | None = None
    """Path to the workspace for file operations"""

    additional_context: dict[str, Any] = field(default_factory=dict)
    """Any additional context specific to this execution"""


@dataclass
class RoleResult:
    """Result of a role execution."""

    success: bool
    """Whether the task completed successfully"""

    output: str
    """Primary output from the role (could be text, JSON, etc.)"""

    artifacts: list[Artifact] = field(default_factory=list)
    """Artifacts produced or modified by this role"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the execution"""

    error: str | None = None
    """Error message if success=False"""


class Role(ABC):
    """
    Base class for all QuestFoundry roles.

    Roles are specialized AI agents that execute domain-specific tasks within
    QuestFoundry's creative workflow. Each role has a specific responsibility
    and communicates via the Layer 4 protocol envelope system.

    Core QuestFoundry roles:
        - Showrunner (SR): Loop coordinator, TU manager, role orchestration
        - Writer (WR): Manuscript content creation, scene writing
        - Archivist (AR): Canon management, worldbuilding consistency
        - Illustrator (IL): Visual asset planning and direction
        - AudioProducer (AP): Audio cue planning and direction
        - Gatekeeper (GK): Quality validation via quality bars
        - Researcher (RS): External research and reference gathering
        - Stylist (ST): Style guide maintenance and voice consistency
        - Editor (ED): Editorial review and refinement

    Role responsibilities:
        1. Load and interpret specialized prompts from spec/01-roles/briefs/
        2. Format execution context from artifacts and project state
        3. Execute tasks via configured LLM provider
        4. Produce structured outputs (artifacts, metadata)
        5. Handle errors gracefully with fallback strategies
        6. Support interactive mode via human callbacks

    Role lifecycle:
        1. Initialization with provider and configuration
        2. Woken by Showrunner or loop with specific task
        3. Loads relevant context (artifacts, canon, project metadata)
        4. Formats prompt with context and task instructions
        5. Executes via LLM provider (with caching/rate limiting)
        6. Parses response into structured artifacts
        7. Returns RoleResult with outputs and metadata

    Key features:
        - Context-aware: Loads relevant artifacts and project state
        - Provider-agnostic: Works with any TextProvider implementation
        - Session tracking: Optional conversation history
        - Human-in-the-loop: Interactive callbacks for clarification
        - Multi-modal: Optional image/audio provider support
        - Configurable: Task-specific and role-level configuration
        - Traceable: Full logging and metadata tracking

    Implementing a custom role:
        >>> from questfoundry.roles.base import Role, RoleContext, RoleResult
        >>> class CustomRole(Role):
        ...     @property
        ...     def role_name(self) -> str:
        ...         return "custom"
        ...
        ...     @property
        ...     def role_description(self) -> str:
        ...         return "Custom role for specific task"
        ...
        ...     def execute_task(self, context: RoleContext) -> RoleResult:
        ...         # Build prompts using base class helpers
        ...         system_prompt = self.build_system_prompt(context)
        ...         user_prompt = self.build_user_prompt(context)
        ...
        ...         # Execute via provider
        ...         response = self.provider.generate_text(
        ...             system_prompt + "\n\n" + user_prompt
        ...         )
        ...
        ...         # Return results (parse artifacts as needed)
        ...         return RoleResult(
        ...             success=True,
        ...             output=response,
        ...             artifacts=[]  # Parse from response if needed
        ...         )

    Example role usage:
        >>> from questfoundry.roles.writer import Writer
        >>> from questfoundry.providers.text.openai import OpenAIProvider
        >>> provider = OpenAIProvider({"api_key": "sk-..."})
        >>> writer = Writer(provider=provider)
        >>> context = RoleContext(
        ...     task="write_scene",
        ...     artifacts=[hook, canon],
        ...     project_metadata={"style": "fantasy"}
        ... )
        >>> result = writer.execute_task(context)
        >>> print(result.output)
        >>> for artifact in result.artifacts:
        ...     print(artifact.type, artifact.artifact_id)
    """

    def __init__(
        self,
        provider: TextProvider,
        spec_path: Path | None = None,
        config: dict[str, Any] | None = None,
        session: "RoleSession | None" = None,
        human_callback: "HumanCallback | None" = None,
        role_config: dict[str, Any] | None = None,
        image_provider: ImageProvider | None = None,
        audio_provider: AudioProvider | None = None,
    ):
        """
        Initialize role with provider and configuration.

        Args:
            provider: Text provider for LLM interactions
            spec_path: Path to spec directory (default: ./spec)
            config: Role-specific configuration (task settings, parameters)
            session: Optional session for conversation history tracking
            human_callback: Optional callback for agent-to-human questions
            role_config: Role-level configuration from global config file
                        (provider selection, cache settings, rate limits)
            image_provider: Optional image generation provider
                           (for roles like Illustrator)
            audio_provider: Optional audio generation provider
                           (for roles like AudioProducer)

        Note:
            The `role_config` parameter contains settings from the global
            configuration file's roles section and is typically used for
            provider selection and global rate limiting/caching settings.

            The `config` parameter is for local, task-specific settings
            and overrides from the application code.

        Example:
            # From configuration file
            role_config = {
                "text_provider": "ollama",
                "cache": {"ttl_seconds": 3600},
                "rate_limit": {"requests_per_minute": 30}
            }

            # Initialize role
            role = PlotWright(
                provider=provider,
                config={"max_tokens": 2000},
                role_config=role_config
            )
        """
        logger.debug("Initializing %s role", self.__class__.__name__)
        logger.trace(
            "Provider: %s, Has image provider: %s, Has audio provider: %s",
            provider.__class__.__name__,
            image_provider is not None,
            audio_provider is not None,
        )

        self.provider = provider
        self.config = config or {}
        self.role_config = role_config or {}
        self.session = session
        self.human_callback = human_callback
        self.image_provider = image_provider
        self.audio_provider = audio_provider

        # Determine spec path
        if spec_path is None:
            # Try to find spec relative to project root
            spec_path = Path.cwd() / "spec"
            if not spec_path.exists():
                # Fall back to relative to this file
                spec_path = Path(__file__).parent.parent.parent.parent / "spec"

        self.spec_path = spec_path
        logger.trace("Spec path set to: %s", self.spec_path)
        self._prompt_cache: dict[str, str] = {}
        logger.info("Role %s initialized successfully", self.__class__.__name__)

    @property
    def has_image_provider(self) -> bool:
        """Check if image generation is available."""
        return self.image_provider is not None

    @property
    def has_audio_provider(self) -> bool:
        """Check if audio generation is available."""
        return self.audio_provider is not None

    @property
    @abstractmethod
    def role_name(self) -> str:
        """
        The role identifier (e.g., 'plotwright', 'gatekeeper').

        This should match the filename in spec/01-roles/briefs/
        """
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable role name (e.g., 'Plotwright', 'Gatekeeper')"""
        pass

    def load_brief(self) -> str:
        """
        Load the role brief from spec/01-roles/briefs/{role_name}.md

        Returns:
            The complete brief content

        Raises:
            FileNotFoundError: If brief file doesn't exist
        """
        logger.trace("Loading brief for role: %s", self.role_name)
        brief_path = self.spec_path / "01-roles" / "briefs" / f"{self.role_name}.md"

        if not brief_path.exists():
            logger.error("Role brief not found at: %s", brief_path)
            raise FileNotFoundError(
                f"Role brief not found: {brief_path}\n"
                f"Expected spec path: {self.spec_path}"
            )

        brief_content = brief_path.read_text(encoding="utf-8")
        logger.debug(
            "Loaded brief for role %s, size: %d bytes",
            self.role_name,
            len(brief_content),
        )
        return brief_content

    def extract_section(self, content: str, section_name: str) -> str:
        """
        Extract a specific section from markdown content.

        Args:
            content: Markdown content
            section_name: Section heading to extract (without #)

        Returns:
            Section content, or empty string if not found
        """
        # Match "## N) Section Name" or "## Section Name"
        pattern = rf"##\s+(?:\d+\))?\s*{re.escape(section_name)}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()
        return ""

    def build_system_prompt(self, context: RoleContext) -> str:
        """
        Build system prompt from role brief and context.

        Default implementation extracts key sections from the brief.
        Subclasses can override for custom prompt construction.

        Args:
            context: Execution context

        Returns:
            System prompt for the LLM
        """
        brief = self.load_brief()

        # Extract key sections
        mindset = self.extract_section(brief, "Mindset") or brief.split("\n")[0]
        principles = self.extract_section(brief, "Operating principles")
        inputs_outputs = self.extract_section(brief, "Inputs & outputs")

        # Build prompt
        prompt_parts = [
            f"You are the {self.display_name} for the QuestFoundry system.",
            "",
            "# Your Mindset",
            mindset.strip(">").strip(),
            "",
        ]

        if principles:
            prompt_parts.extend(
                [
                    "# Operating Principles",
                    principles,
                    "",
                ]
            )

        if inputs_outputs:
            prompt_parts.extend(
                [
                    "# Inputs & Outputs",
                    inputs_outputs,
                    "",
                ]
            )

        return "\n".join(prompt_parts)

    def format_artifacts(self, artifacts: list[Artifact]) -> str:
        """
        Format artifacts for inclusion in prompt context.

        Limits the number of artifacts and total context size to prevent
        resource exhaustion attacks.

        Args:
            artifacts: List of artifacts to format

        Returns:
            Formatted string representation
        """
        if not artifacts:
            return "No artifacts provided."

        # Limit number of artifacts to prevent excessive memory usage
        artifacts_to_format = artifacts[:MAX_ARTIFACTS_IN_CONTEXT]
        truncated_count = max(0, len(artifacts) - MAX_ARTIFACTS_IN_CONTEXT)

        header_text = "# Available Artifacts\n"
        formatted = [header_text]
        total_size = len(header_text)

        for artifact in artifacts_to_format:
            # Get title from data.header.short_name or metadata
            title = "Unknown"
            if isinstance(artifact.data, dict):
                has_header = "header" in artifact.data and isinstance(
                    artifact.data["header"], dict
                )
                if has_header:
                    title = artifact.data["header"].get("short_name", "Unknown")
                elif "title" in artifact.data:
                    title = artifact.data.get("title", "Unknown")

            artifact_parts = [f"## {artifact.type}: {title}"]

            # Get ID from metadata
            artifact_id = artifact.artifact_id or "no-id"
            artifact_parts.append(f"ID: {artifact_id}")

            # Include key fields from data
            if artifact.data:
                artifact_parts.append("\nData:")
                for key, value in artifact.data.items():
                    if key not in ("id", "title"):
                        # Truncate long values
                        str_value = str(value)
                        if len(str_value) > MAX_ARTIFACT_VALUE_LENGTH:
                            str_value = str_value[:MAX_ARTIFACT_VALUE_LENGTH] + "..."
                        artifact_parts.append(f"  {key}: {str_value}")

            artifact_parts.append("")  # Blank line between artifacts
            artifact_text = "\n".join(artifact_parts)
            artifact_text_len = len(artifact_text)

            # Check if adding this artifact would exceed total size limit
            if total_size + artifact_text_len > MAX_FORMATTED_CONTEXT_SIZE:
                formatted.append("\n[Additional artifacts omitted due to size limits]")
                break

            formatted.append(artifact_text)
            total_size += artifact_text_len

        # Add notice if artifacts were truncated
        if truncated_count > 0:
            formatted.append(
                f"\n[Note: {truncated_count} additional artifact(s) omitted. "
                f"Maximum is {MAX_ARTIFACTS_IN_CONTEXT} artifacts.]"
            )

        return "\n".join(formatted)

    def build_user_prompt(self, context: RoleContext) -> str:
        """
        Build user prompt from execution context.

        Args:
            context: Execution context

        Returns:
            User prompt for the LLM
        """
        parts = [
            f"# Task: {context.task}",
            "",
        ]

        # Add artifacts
        if context.artifacts:
            parts.append(self.format_artifacts(context.artifacts))
            parts.append("")

        # Add project metadata
        if context.project_metadata:
            parts.append("# Project Information")
            for key, value in context.project_metadata.items():
                parts.append(f"{key}: {value}")
            parts.append("")

        # Add additional context
        if context.additional_context:
            parts.append("# Additional Context")
            for key, value in context.additional_context.items():
                parts.append(f"{key}: {value}")
            parts.append("")

        parts.append("Please complete the requested task according to your role.")

        return "\n".join(parts)

    @abstractmethod
    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a role-specific task.

        This is the main entry point for role execution. Implementations
        should:
        1. Build prompts from context
        2. Call LLM provider
        3. Parse and validate output
        4. Return structured result

        Args:
            context: Execution context

        Returns:
            Result of task execution
        """
        pass

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Call the LLM provider with prompts.

        Args:
            system_prompt: System/role prompt
            user_prompt: User/task prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        logger.debug("Calling LLM for role %s", self.__class__.__name__)

        # Combine system and user prompts
        # Most providers expect this format
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        effective_max_tokens = max_tokens or self.config.get("max_tokens", 2000)
        effective_temperature = temperature or self.config.get("temperature", 0.7)

        logger.trace(
            (
                "LLM call parameters - max_tokens: %d, "
                "temperature: %.2f, prompt_length: %d"
            ),
            effective_max_tokens,
            effective_temperature,
            len(full_prompt),
        )

        response = self.provider.generate_text(
            prompt=full_prompt,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
        )

        logger.debug(
            "LLM call completed, response length: %d characters", len(response)
        )
        return response

    def _parse_json_from_response(self, response: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        LLMs often wrap JSON in markdown code blocks like ```json or ```.
        This method extracts the JSON content and parses it.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If response doesn't contain valid JSON
        """
        import json

        # Try to extract JSON from markdown code blocks
        json_match = response
        if "```json" in response:
            # Extract content between ```json and ```
            json_match = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            # Extract content between ``` and ```
            json_match = response.split("```")[1].split("```")[0]

        result: dict[str, Any] = json.loads(json_match.strip())
        return result

    def ask_human(
        self,
        question: str,
        context: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> str:
        """
        Ask human a question (interactive mode).

        If no human_callback is configured, returns empty string or first
        suggestion (batch mode).

        Args:
            question: Question to ask
            context: Optional domain-specific context
            suggestions: Optional list of suggested answers
            artifacts: Optional list of relevant artifacts

        Returns:
            Human's response or default answer

        Example:
            >>> answer = role.ask_human(
            ...     "What tone should this scene have?",
            ...     suggestions=["dark", "lighthearted", "neutral"]
            ... )
        """
        logger.info("Asking human: %s", question)
        logger.trace("Number of suggestions: %d", len(suggestions or []))

        callback = self.human_callback or batch_mode_callback
        is_interactive = self.human_callback is not None
        logger.debug(
            "Using %s mode for human callback",
            "interactive" if is_interactive else "batch",
        )

        # Build callback context
        callback_context: dict[str, Any] = {
            "question": question,
            "context": context or {},
            "suggestions": suggestions or [],
            "artifacts": artifacts or [],
            "role": self.role_name,
        }

        response = callback(question, callback_context)
        logger.debug("Human response received: %s", response)
        return response

    def ask_yes_no(
        self,
        question: str,
        default: bool = True,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Ask a yes/no question.

        Args:
            question: Question to ask
            default: Default answer if in batch mode or unclear response
            context: Optional context

        Returns:
            True for yes, False for no

        Example:
            >>> if role.ask_yes_no("Generate images for this scene?"):
            ...     generate_images()
        """
        logger.trace("Asking yes/no question: %s (default: %s)", question, default)
        response = self.ask_human(
            question,
            context=context,
            suggestions=["yes", "no"],
        )

        # Parse response
        response_lower = response.lower().strip()

        if response_lower in ["yes", "y", "true", "1"]:
            logger.debug("Yes/no question answered: true")
            return True
        elif response_lower in ["no", "n", "false", "0"]:
            logger.debug("Yes/no question answered: false")
            return False
        else:
            # If unclear, use default
            logger.debug(
                "Unclear yes/no response '%s', using default: %s", response, default
            )
            return default

    def ask_choice(
        self,
        question: str,
        choices: list[str],
        default: int = 0,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Ask human to choose from a list of options.

        Args:
            question: Question to ask
            choices: List of choices
            default: Index of default choice (0-based)
            context: Optional context

        Returns:
            Selected choice

        Example:
            >>> tone = role.ask_choice(
            ...     "Select scene tone:",
            ...     ["dark", "lighthearted", "neutral"]
            ... )
        """
        logger.trace(
            "Asking choice question: %s with %d options", question, len(choices)
        )
        response = self.ask_human(
            question,
            context=context,
            suggestions=choices,
        )

        # If response matches a choice, return it
        if response in choices:
            logger.debug("Choice selected: %s", response)
            return response

        # Try to parse as number (1-indexed)
        try:
            index = int(response) - 1
            if 0 <= index < len(choices):
                selected = choices[index]
                logger.debug("Choice selected by index: %d -> %s", index + 1, selected)
                return selected
        except ValueError:
            # If input is not a valid integer, fall back to default below
            logger.trace("Could not parse choice as integer: %s", response)
            pass

        # Fall back to default with bounds checking
        if 0 <= default < len(choices):
            selected = choices[default]
            logger.debug("Using default choice: %d -> %s", default, selected)
            return selected

        fallback = choices[0] if choices else ""
        logger.warning("No valid choice, using fallback: %s", fallback)
        return fallback

    def __repr__(self) -> str:
        """String representation of the role."""
        session_info = f", session={bool(self.session)}" if self.session else ""
        callback_info = (
            f", interactive={bool(self.human_callback)}" if self.human_callback else ""
        )
        return (
            f"{self.__class__.__name__}("
            f"role_name='{self.role_name}'"
            f"{session_info}{callback_info})"
        )
