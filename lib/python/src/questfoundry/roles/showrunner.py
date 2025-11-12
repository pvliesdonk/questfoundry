"""Showrunner role implementation."""

import logging
from typing import TYPE_CHECKING, Any

from .base import Role, RoleContext, RoleResult

if TYPE_CHECKING:
    from questfoundry.providers.base import ImageProvider, TextProvider
    from questfoundry.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class Showrunner(Role):
    """
    Showrunner: Loop coordinator and TU manager.

    The Showrunner orchestrates the workflow by selecting appropriate loops,
    waking the minimal set of roles needed, and managing TU progression from
    planning through merge. They keep momentum high with small, focused changes.

    Key responsibilities:
    - Select appropriate loop based on project state and goals
    - Open TUs with clear scope and deliverables
    - Wake only necessary roles for the TU
    - Coordinate between roles and manage handoffs
    - Ensure pre-gate checks before merges
    - Manage snapshots and view generation
    - File hooks for out-of-scope work
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/showrunner.md"""
        return "showrunner"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Showrunner"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a showrunner task.

        Supported tasks:
        - 'select_loop': Choose appropriate loop for current goals
        - 'open_tu': Create TU brief with scope and deliverables
        - 'plan_roles': Determine which roles to wake
        - 'coordinate_step': Manage handoff between roles
        - 'review_progress': Check TU progress and next steps
        - 'collect_metrics': Collect project metrics (alias for review_progress)
        - 'create_snapshot': Create project snapshot (alias for coordinate_step)

        Args:
            context: Execution context

        Returns:
            Result with coordination decisions
        """
        task = context.task.lower()

        if task == "select_loop":
            return self._select_loop(context)
        elif task == "open_tu":
            return self._open_tu(context)
        elif task == "plan_roles":
            return self._plan_roles(context)
        elif task == "coordinate_step":
            return self._coordinate_step(context)
        elif task == "review_progress":
            return self._review_progress(context)
        # New tasks for loops
        elif task == "collect_metrics":
            return self._review_progress(context)
        elif task == "create_snapshot":
            return self._coordinate_step(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _select_loop(self, context: RoleContext) -> RoleResult:
        """Select appropriate loop based on project state and goals."""
        system_prompt = self.build_system_prompt(context)

        # Get available loops from context
        available_loops = context.additional_context.get("available_loops", [])
        current_goal = context.additional_context.get("goal", "")
        project_state = context.additional_context.get("project_state", {})

        user_prompt = f"""# Task: Select Loop

{self.format_artifacts(context.artifacts)}

## Current Goal
{current_goal}

## Project State
{self._format_dict(project_state)}

## Available Loops
{self._format_loops(available_loops)}

Based on the current goal and project state, select the most appropriate loop to run.
Consider:
- What needs to be created or fixed?
- What roles are required?
- How much time is available?
- What quality bars are most at risk?

Respond with:
- **Selected Loop**: The loop_id to run
- **Rationale**: Why this loop fits the current need
- **Estimated Duration**: Expected time commitment
- **Roles to Wake**: Which roles are needed
- **Bar Pressure**: Which quality bars will be stressed
- **Risks**: Potential issues to watch for
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "loop_selection"},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error selecting loop: {e}",
            )

    def _open_tu(self, context: RoleContext) -> RoleResult:
        """Create TU brief with scope and deliverables."""
        system_prompt = self.build_system_prompt(context)

        loop_id = context.additional_context.get("loop_id", "")
        goal = context.additional_context.get("goal", "")
        roles_awake = context.additional_context.get("roles_awake", [])

        user_prompt = f"""# Task: Open TU

{self.format_artifacts(context.artifacts)}

## Goal
{goal}

## Selected Loop
{loop_id}

## Roles Awake
{", ".join(roles_awake)}

Create a TU brief that defines the scope, deliverables, and risks for this work unit.

Include:
1. **TU ID**: Short identifier (e.g., "hook-refinement-act1")
2. **Slice**: Specific portion being worked on
3. **Loop**: Which loop will be run
4. **Roles Awake**: Who's actively working
5. **Deliverables**: Concrete outputs expected
6. **Bar Pressure**: Which quality bars will be stressed
7. **Risks**: Potential issues (dependencies, dormant roles, etc.)
8. **Timebox**: Estimated duration

Keep it concise - one paragraph or bulleted list.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "tu_brief"},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error opening TU: {e}",
            )

    def _plan_roles(self, context: RoleContext) -> RoleResult:
        """Determine which roles to wake for the TU."""
        system_prompt = self.build_system_prompt(context)

        loop_id = context.additional_context.get("loop_id", "")
        goal = context.additional_context.get("goal", "")

        user_prompt = f"""# Task: Plan Roles

## Goal
{goal}

## Selected Loop
{loop_id}

Determine which roles need to be "awake" (actively working) for this TU.

Consider:
- What work needs to be done?
- Which domains are affected (structure, prose, world, terms, assets)?
- Are any roles dormant (no recent work in their domain)?
- Can we keep the role set minimal?

For each role, indicate:
- **Awake**: Actively working on deliverables
- **Consulted**: Providing input but not primary owner
- **Dormant**: Not needed for this TU

Roles available: plotwright, scene_smith, style_lead, lore_weaver,
codex_curator, gatekeeper, art_director, audio_engineer, translator
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "role_plan"},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error planning roles: {e}",
            )

    def _coordinate_step(self, context: RoleContext) -> RoleResult:
        """Manage handoff between roles in a loop step."""
        system_prompt = self.build_system_prompt(context)

        step_name = context.additional_context.get("step_name", "")
        from_role = context.additional_context.get("from_role", "")
        to_role = context.additional_context.get("to_role", "")

        user_prompt = f"""# Task: Coordinate Step

{self.format_artifacts(context.artifacts)}

## Current Step
{step_name}

## Handoff
From: {from_role} → To: {to_role}

The previous step has completed. Review the artifacts and provide coordination
notes for the next step:

1. **What's Ready**: Artifacts/outputs from previous step
2. **Context for Next Role**: Key information to carry forward
3. **Open Questions**: Anything that needs clarification
4. **Risks**: Issues to watch for in the next step

Keep it brief and actionable.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1000)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "coordination"},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error coordinating step: {e}",
            )

    def _review_progress(self, context: RoleContext) -> RoleResult:
        """Review TU progress and determine next steps."""
        system_prompt = self.build_system_prompt(context)

        tu_id = context.additional_context.get("tu_id", "")
        steps_completed = context.additional_context.get("steps_completed", [])
        steps_remaining = context.additional_context.get("steps_remaining", [])

        user_prompt = f"""# Task: Review Progress

{self.format_artifacts(context.artifacts)}

## TU: {tu_id}

## Steps Completed
{self._format_list(steps_completed)}

## Steps Remaining
{self._format_list(steps_remaining)}

Review the TU progress and provide:

1. **Status Summary**: Where we are in the TU
2. **Quality Check**: Are we meeting quality bars so far?
3. **Blockers**: Any issues preventing progress?
4. **Next Step**: What should happen next (one line)
5. **Hooks Filed**: Any out-of-scope ideas to capture

Be concise and action-oriented.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "progress_review"},
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error reviewing progress: {e}",
            )

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format dictionary as bullet list."""
        if not d:
            return "(empty)"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())

    def _format_list(self, items: list[str]) -> str:
        """Format list as bullet list."""
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)

    def _format_loops(self, loops: list[dict[str, Any]]) -> str:
        """Format loop metadata list."""
        if not loops:
            return "(no loops available)"

        formatted = []
        for loop in loops:
            formatted.append(
                f"- **{loop.get('loop_id')}**: {loop.get('description')} "
                f"({loop.get('typical_duration')})"
            )
        return "\n".join(formatted)

    def get_provider_for_role(
        self,
        registry: "ProviderRegistry",
        provider_type: str = "text",
    ) -> "TextProvider | ImageProvider":
        """
        Get the appropriate provider for this role based on configuration.

        Uses role-specific configuration to determine which provider instance
        to use. Falls back to default if no role-specific configuration exists.

        Args:
            registry: ProviderRegistry instance with all configured providers
            provider_type: Type of provider ('text' or 'image')

        Returns:
            Provider instance configured for this role

        Raises:
            ValueError: If provider cannot be initialized

        Example:
            registry = ProviderRegistry(config)
            provider = showrunner.get_provider_for_role(registry, "text")
        """
        # Get the provider name for this role
        provider_name = registry.config.get_role_provider(self.role_name, provider_type)

        # Use registry to get/create the provider instance
        if provider_type == "text":
            return registry.get_text_provider(provider_name)
        elif provider_type == "image":
            return registry.get_image_provider(provider_name)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def initialize_role_with_config(
        self,
        role_class: type[Role],
        registry: "ProviderRegistry",
        spec_path: Any | None = None,
        config: dict[str, Any] | None = None,
        session: Any | None = None,
        human_callback: Any | None = None,
        role_name: str | None = None,
    ) -> Role:
        """
        Initialize a role with provider configuration from global config.

        This method creates a role instance using the appropriate provider
        based on role-specific configuration. It handles:
        - Provider selection (per-role vs default)
        - Cache and rate limiting configuration
        - Per-role provider instantiation

        Args:
            role_class: Role class to instantiate
            registry: ProviderRegistry with all providers and configuration
            spec_path: Path to spec directory (optional)
            config: Task-specific configuration (optional)
            session: RoleSession for conversation tracking (optional)
            human_callback: Callback for human interaction (optional)
            role_name: Role name for config lookup (optional, auto-derived if not
                provided)

        Returns:
            Initialized role instance ready for use

        Raises:
            ValueError: If provider cannot be initialized

        Example:
            config = ProviderConfig()
            registry = ProviderRegistry(config)
            plotwright = showrunner.initialize_role_with_config(
                PlotWright,
                registry,
                config={"max_tokens": 2000}
            )
        """
        try:
            # Get role_name - either from parameter or derive from class
            if role_name is None:
                # Create minimal instance with default provider to get role_name
                default_provider = registry.get_text_provider()
                temp_role = role_class(provider=default_provider)
                role_name = temp_role.role_name

            # Get the actual provider for this role
            provider_name = registry.config.get_role_provider(role_name, "text")
            provider = registry.get_text_provider(provider_name)

            # Get role-specific configuration
            role_config = registry.config.get_role_config(role_name)

        except Exception as e:
            raise ValueError(f"Failed to initialize provider for role: {e}") from e

        # Create and return the role with proper configuration
        return role_class(
            provider=provider,
            spec_path=spec_path,
            config=config,
            session=session,
            human_callback=human_callback,
            role_config=role_config,
        )
