"""Base classes for QuestFoundry loops."""

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role
from ..state.workspace import WorkspaceManager
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a loop step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LoopStep:
    """
    Single step in a loop execution.

    Each step represents a discrete task performed by one or more roles.
    """

    step_id: str
    """Unique identifier for this step"""

    description: str
    """Human-readable description of what this step does"""

    assigned_roles: list[str] = field(default_factory=list)
    """RACI: Responsible roles that perform this step"""

    consulted_roles: list[str] = field(default_factory=list)
    """RACI: Consulted roles that provide input"""

    informed_roles: list[str] = field(default_factory=list)
    """RACI: Informed roles that receive updates"""

    artifacts_input: list[str] = field(default_factory=list)
    """Required artifact types for input"""

    artifacts_output: list[str] = field(default_factory=list)
    """Expected artifact types to produce"""

    validation_required: bool = True
    """Whether this step requires validation before proceeding"""

    status: StepStatus = StepStatus.PENDING
    """Current status of this step"""

    result: Any | None = None
    """Result of executing this step"""

    error: str | None = None
    """Error message if step failed"""


@dataclass
class LoopContext:
    """
    Context for active loop execution.

    This contains all information needed for a loop to execute,
    approximately ~500 lines when formatted for LLM context.
    """

    loop_id: str
    """ID of the loop being executed"""

    project_id: str
    """ID of the project"""

    workspace: WorkspaceManager
    """Workspace for artifact storage"""

    role_instances: dict[str, Role] = field(default_factory=dict)
    """Instantiated role objects keyed by role name"""

    artifacts: list[Artifact] = field(default_factory=list)
    """Artifacts available for this loop"""

    project_metadata: dict[str, Any] = field(default_factory=dict)
    """Project-level metadata"""

    current_step: int = 0
    """Index of currently executing step"""

    history: list[dict[str, Any]] = field(default_factory=list)
    """Execution history"""

    config: dict[str, Any] = field(default_factory=dict)
    """Loop-specific configuration"""


@dataclass
class LoopResult:
    """Result of loop execution."""

    success: bool
    """Whether the loop completed successfully"""

    loop_id: str
    """ID of the executed loop"""

    artifacts_created: list[Artifact] = field(default_factory=list)
    """Artifacts created during execution"""

    artifacts_modified: list[Artifact] = field(default_factory=list)
    """Artifacts modified during execution"""

    steps_completed: int = 0
    """Number of steps completed"""

    steps_failed: int = 0
    """Number of steps that failed"""

    error: str | None = None
    """Error message if loop failed"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about execution"""


class Loop(ABC):
    """
    Base class for all QuestFoundry loop implementations.

    Loops are multi-step workflows that orchestrate role execution for specific
    creative objectives. Unlike runtime playbook parsing, loops are hardcoded
    Python classes with explicit step definitions, reducing LLM context usage
    while maintaining flexibility through LLM-backed role execution.

    Core loop types:
        - Scene Forge: Manuscript scene development
        - Hook Harvest: Identify and classify new hooks
        - Lore Deepening: Expand canon and worldbuilding
        - Gatecheck: Quality validation before cold promotion
        - Archive Snapshot: Create cold storage snapshots
        - Art Touch-Up: Visual asset refinement
        - Audio Pass: Audio production and cue implementation
        - Style Tune-Up: Voice and style consistency improvements
        - Codex Expansion: Player-facing reference expansion
        - Binding Run: Final assembly and export

    Loop architecture:
        - Fixed step sequence defined in Python
        - Each step assigns specific roles (RACI model)
        - Roles execute with LLM autonomy within step scope
        - Artifacts flow between steps
        - Validation gates control progression
        - Context limited to ~500 lines for efficiency

    Step execution flow:
        1. Load step definition (assigned roles, inputs, outputs)
        2. Wake assigned roles with step context
        3. Roles execute and produce artifacts
        4. Validate outputs (if validation_required)
        5. Update loop context and history
        6. Proceed to next step or terminate

    Benefits of hardcoded loops:
        - Predictable execution paths
        - Reduced LLM context requirements
        - Clear role responsibility boundaries
        - Type-safe step definitions
        - Easy testing and debugging
        - Explicit artifact flow
        - Compile-time step validation

    Example loop structure:
        >>> class CustomLoop(Loop):
        ...     @property
        ...     def loop_name(self) -> str:
        ...         return "custom_loop"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Custom workflow for specific task"
        ...
        ...     def define_steps(self) -> list[LoopStep]:
        ...         return [
        ...             LoopStep(
        ...                 step_id="step_1",
        ...                 description="Gather requirements",
        ...                 assigned_roles=["researcher"],
        ...                 artifacts_input=["hook_card"],
        ...                 artifacts_output=["research_memo"]
        ...             ),
        ...             LoopStep(
        ...                 step_id="step_2",
        ...                 description="Create content",
        ...                 assigned_roles=["writer"],
        ...                 artifacts_input=["research_memo"],
        ...                 artifacts_output=["manuscript_section"]
        ...             )
        ...         ]
        ...
        ...     def execute(self, context: LoopContext) -> LoopResult:
        ...         # Execute steps sequentially
        ...         for step in self.steps:
        ...             result = self.execute_step(step, context)
        ...             if not result.success:
        ...                 return LoopResult(
        ...                     success=False,
        ...                     loop_id=self.loop_name,
        ...                     error=result.error
        ...                 )
        ...         return LoopResult(success=True, loop_id=self.loop_name)

    Loop execution example:
        >>> from questfoundry.loops.scene_forge import SceneForgeLoop
        >>> # Assuming workspace, writer, and archivist are pre-instantiated
        >>> loop = SceneForgeLoop()
        >>> context = LoopContext(
        ...     loop_id="scene_forge",
        ...     project_id="my_project",
        ...     workspace=workspace,
        ...     role_instances={"writer": writer, "archivist": archivist}
        ... )
        >>> result = loop.execute(context)
        >>> print(f"Created {len(result.artifacts_created)} artifacts")
    """

    # Class-level metadata (defined by subclasses)
    metadata: LoopMetadata

    # Steps for this loop (defined by subclasses)
    steps: list[LoopStep] = []

    def __init__(self, context: LoopContext):
        """
        Initialize loop with execution context.

        Args:
            context: Loop execution context
        """
        logger.info("Initializing %s loop", self.__class__.__name__)
        logger.trace(
            "Loop ID: %s, Project ID: %s, Initial step: %d",
            context.loop_id,
            context.project_id,
            context.current_step,
        )

        self.context = context
        self.current_step_index = context.current_step
        # Create instance-specific copy of steps to avoid shared state
        self.steps = copy.deepcopy(self.__class__.steps)

        logger.debug(
            "Loop %s initialized with %d steps",
            self.__class__.__name__,
            len(self.steps),
        )

    @abstractmethod
    def execute(self) -> LoopResult:
        """
        Execute the complete loop.

        This is the main entry point for loop execution. Implementations
        should:
        1. Iterate through steps
        2. Execute each step with appropriate roles
        3. Validate outputs
        4. Handle failures
        5. Return result

        Returns:
            Result of loop execution
        """
        pass

    def execute_step(self, step: LoopStep) -> None:
        """
        Execute a single step.

        Default implementation:
        1. Mark step as in_progress
        2. Get required roles
        3. Execute role tasks
        4. Validate if required
        5. Mark step completed or failed

        Subclasses can override for custom behavior.

        Args:
            step: Step to execute

        Raises:
            ValueError: If required roles not available
        """
        logger.info("Executing step: %s", step.step_id)
        logger.debug("Step description: %s", step.description)
        logger.trace("Assigned roles: %s", ", ".join(step.assigned_roles))

        step.status = StepStatus.IN_PROGRESS

        # Get assigned roles
        roles_needed = step.assigned_roles
        roles = {}
        for role_name in roles_needed:
            if role_name not in self.context.role_instances:
                logger.error("Required role not available: %s", role_name)
                step.status = StepStatus.FAILED
                step.error = f"Required role '{role_name}' not available"
                raise ValueError(step.error)
            roles[role_name] = self.context.role_instances[role_name]

        logger.debug("All required roles available for step: %s", step.step_id)

        # Execute (subclass implements specific logic)
        try:
            logger.trace("Calling step logic for: %s", step.step_id)
            result = self._execute_step_logic(step, roles)
            step.result = result
            logger.debug("Step logic completed for: %s", step.step_id)

            # Validate if required
            if step.validation_required:
                logger.trace("Validating step: %s", step.step_id)
                is_valid = self.validate_step(step, result)
                if not is_valid:
                    logger.warning("Step validation failed: %s", step.step_id)
                    step.status = StepStatus.FAILED
                    step.error = "Validation failed"
                else:
                    logger.info("Step completed successfully: %s", step.step_id)
                    step.status = StepStatus.COMPLETED
            else:
                logger.info("Step completed (no validation required): %s", step.step_id)
                step.status = StepStatus.COMPLETED

        except Exception as e:
            logger.error("Error executing step %s: %s", step.step_id, e, exc_info=True)
            step.status = StepStatus.FAILED
            step.error = str(e)
            raise

    def _execute_step_logic(self, step: LoopStep, roles: dict[str, Role]) -> Any:
        """
        Execute the actual step logic.

        Subclasses should override this to implement step-specific behavior.

        Args:
            step: Step being executed
            roles: Available roles

        Returns:
            Step result
        """
        # Default: just return empty dict
        # Subclasses override for actual implementation
        return {}

    def validate_step(self, step: LoopStep, result: Any) -> bool:
        """
        Validate step completion.

        Default implementation always returns True.
        Subclasses can override for specific validation logic.

        Args:
            step: Step that was executed
            result: Result from step execution

        Returns:
            True if step is valid, False otherwise
        """
        return True

    def can_continue(self) -> bool:
        """
        Check if loop can proceed to next step.

        Returns:
            True if can continue, False otherwise
        """
        logger.trace("Checking if loop can continue")

        # Check if there are more steps
        if self.current_step_index >= len(self.steps):
            logger.debug(
                "No more steps available (current: %d, total: %d)",
                self.current_step_index,
                len(self.steps),
            )
            return False

        # Check if previous step succeeded
        if self.current_step_index > 0:
            prev_step = self.steps[self.current_step_index - 1]
            if prev_step.status == StepStatus.FAILED:
                logger.warning("Previous step failed, cannot continue")
                return False

        logger.trace("Loop can continue to next step")
        return True

    def rollback_step(self) -> None:
        """
        Roll back to previous step.

        Default implementation just decrements step index.
        Subclasses can override for cleanup logic.
        """
        logger.warning("Rolling back loop to previous step")

        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.context.current_step = self.current_step_index

            # Mark current step as pending
            if self.current_step_index < len(self.steps):
                step_name = self.steps[self.current_step_index].step_id
                self.steps[self.current_step_index].status = StepStatus.PENDING
                logger.info("Rolled back to step: %s", step_name)
        else:
            logger.debug("Cannot rollback - already at first step")

    def skip_step(self, step: LoopStep) -> None:
        """
        Skip a step (mark as skipped and move on).

        Args:
            step: Step to skip
        """
        logger.info("Skipping step: %s", step.step_id)
        logger.debug("Step description: %s", step.description)

        step.status = StepStatus.SKIPPED
        self.current_step_index += 1
        self.context.current_step = self.current_step_index

        logger.trace("Advanced to next step index: %d", self.current_step_index)

    def build_loop_context_summary(self) -> str:
        """
        Build detailed context for loop execution.

        This creates approximately ~500 lines of context including:
        - Loop metadata
        - Available artifacts
        - Step definitions
        - Role assignments
        - Quality gates

        Returns:
            Formatted context string
        """
        lines = [
            f"# Loop: {self.metadata.display_name}",
            "",
            f"**Purpose**: {self.metadata.description}",
            f"**Duration**: {self.metadata.typical_duration}",
            "",
            "## Steps",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. **{step.description}**")
            lines.append(f"   - Assigned: {', '.join(step.assigned_roles)}")
            if step.consulted_roles:
                lines.append(f"   - Consulted: {', '.join(step.consulted_roles)}")
            if step.artifacts_input:
                lines.append(f"   - Input: {', '.join(step.artifacts_input)}")
            if step.artifacts_output:
                lines.append(f"   - Output: {', '.join(step.artifacts_output)}")
            lines.append("")

        lines.append("## Available Artifacts")
        lines.append("")
        for artifact in self.context.artifacts:
            lines.append(f"- {artifact.type}: {artifact.artifact_id}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the loop."""
        return (
            f"{self.__class__.__name__}("
            f"loop_id='{self.metadata.loop_id}', "
            f"step={self.current_step_index}/{len(self.steps)})"
        )
