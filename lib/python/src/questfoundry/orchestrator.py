"""Orchestrator for QuestFoundry loop execution."""

import re
from pathlib import Path
from typing import Any, cast

from .logging_config import get_logger
from .loops.base import Loop, LoopContext, LoopResult
from .loops.registry import LoopRegistry
from .models.artifact import Artifact
from .providers.base import TextProvider
from .providers.config import ProviderConfig
from .providers.registry import ProviderRegistry
from .roles.base import RoleContext
from .roles.registry import RoleRegistry
from .roles.showrunner import Showrunner
from .state.workspace import WorkspaceManager

logger = get_logger(__name__)


class Orchestrator:
    """
    Orchestrates QuestFoundry workflow execution.

    The Orchestrator coordinates between the Showrunner role, loop registry,
    and role registry to execute appropriate loops based on user goals.

    It manages:
    - Loop selection based on project state and goals
    - Role instantiation and lifecycle
    - Loop execution and coordination
    - Artifact management
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        provider_registry: ProviderRegistry | None = None,
        role_registry: RoleRegistry | None = None,
        loop_registry: LoopRegistry | None = None,
        spec_path: Path | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            workspace: Workspace manager for project
            provider_registry: Provider registry (creates default if None)
            role_registry: Role registry (creates default if None)
            loop_registry: Loop registry (creates default if None)
            spec_path: Path to spec directory
        """
        logger.debug("Initializing Orchestrator")
        self.workspace = workspace
        self.spec_path = spec_path or Path.cwd() / "spec"

        logger.trace("Orchestrator spec_path=%s", self.spec_path)

        # Initialize registries
        self.provider_registry = provider_registry or ProviderRegistry(
            config=ProviderConfig()
        )
        self.role_registry = role_registry or RoleRegistry(
            self.provider_registry, spec_path=self.spec_path
        )
        self.loop_registry = loop_registry or LoopRegistry(spec_path=self.spec_path)

        logger.trace(
            "Registries initialized - roles=%d, loops=%d",
            len(self.role_registry.list_roles()),
            len(self.loop_registry.list_loops()),
        )

        # Initialize showrunner and provider (type annotations for mypy)
        self.showrunner: Showrunner | None = None
        self.provider: TextProvider | None = None
        self.provider_name: str | None = None

        logger.info("Orchestrator initialized successfully")

    def initialize(
        self,
        provider: TextProvider | None = None,
        provider_name: str | None = None,
    ) -> None:
        """
        Initialize orchestrator with LLM provider.

        Args:
            provider: Text provider to use
            provider_name: Name of provider in registry
        """
        logger.debug("Initializing orchestrator with provider_name=%s", provider_name)

        # Save provider for later use
        self.provider = provider
        self.provider_name = provider_name

        logger.trace("Getting showrunner role instance")
        # Get showrunner instance
        self.showrunner = cast(
            Showrunner,
            self.role_registry.get_role(
                "showrunner",
                provider=provider,
                provider_name=provider_name,
            ),
        )

        logger.info("Orchestrator initialized with provider '%s'", provider_name)

    def select_loop(
        self,
        goal: str,
        project_state: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> str:
        """
        Select appropriate loop for the given goal.

        Args:
            goal: User's goal or request
            project_state: Current project state
            artifacts: Existing artifacts

        Returns:
            Loop ID to execute

        Raises:
            RuntimeError: If showrunner not initialized or selection fails
        """
        logger.info("Selecting loop for goal: %s", goal)

        if self.showrunner is None:
            logger.error("Orchestrator not initialized - showrunner is None")
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        # Get available loops
        loops = self.loop_registry.list_loops()
        logger.debug("Retrieved %d available loops", len(loops))
        logger.trace("Available loops: %s", [m.loop_id for m in loops])

        loop_data = [
            {
                "loop_id": metadata.loop_id,
                "description": metadata.description,
                "typical_duration": metadata.typical_duration,
                "primary_roles": metadata.primary_roles,
                "entry_conditions": metadata.entry_conditions,
            }
            for metadata in loops
        ]

        # Create context for showrunner
        project_info = self.workspace.get_project_info()
        logger.trace(
            "Creating RoleContext for showrunner task with %d artifacts",
            len(artifacts or []),
        )

        context = RoleContext(
            task="select_loop",
            artifacts=artifacts or [],
            project_metadata=project_info.model_dump(),
            additional_context={
                "goal": goal,
                "project_state": project_state or {},
                "available_loops": loop_data,
            },
        )

        # Call showrunner to select loop
        logger.trace("Calling showrunner.execute_task()")
        result = self.showrunner.execute_task(context)

        if not result.success:
            logger.error("Loop selection failed: %s", result.error)
            raise RuntimeError(f"Loop selection failed: {result.error}")

        # Parse loop_id from response
        # For now, extract from output (in production, would parse structured response)
        logger.trace("Extracting loop ID from showrunner output")
        loop_id = self._extract_loop_id(result.output)

        logger.info("Selected loop '%s' for goal '%s'", loop_id, goal)
        return loop_id

    def execute_loop(
        self,
        loop_id: str,
        project_id: str,
        artifacts: list[Artifact] | None = None,
        config: dict[str, Any] | None = None,
    ) -> LoopResult:
        """
        Execute a specific loop.

        Args:
            loop_id: Loop identifier
            project_id: Project identifier
            artifacts: Existing artifacts
            config: Loop configuration

        Returns:
            Loop execution result

        Raises:
            KeyError: If loop not found
            RuntimeError: If required roles not available
        """
        logger.info("Executing loop '%s' for project '%s'", loop_id, project_id)

        # Get loop metadata (raises KeyError if not found)
        logger.debug("Retrieving loop metadata for loop '%s'", loop_id)
        metadata = self.loop_registry.get_loop_metadata(loop_id)
        logger.trace(
            "Loop metadata retrieved - primary_roles=%d, consulted_roles=%d",
            len(metadata.primary_roles),
            len(metadata.consulted_roles),
        )

        # Extract human_callback from config
        human_callback = (config or {}).get("human_callback")
        if human_callback:
            logger.debug("Interactive mode enabled - human_callback provided")

        # Instantiate required roles
        role_instances = {}
        required_roles = set(metadata.primary_roles + metadata.consulted_roles)
        logger.debug("Instantiating %d required roles", len(required_roles))

        for role_name in required_roles:
            try:
                if human_callback:
                    if self.showrunner is None:
                        raise RuntimeError(
                            "Orchestrator not initialized. Call initialize() first."
                        )
                    # Interactive mode: Use Showrunner to create a new, non-cached
                    # role instance with the callback.
                    logger.trace(
                        "Instantiating role '%s' via Showrunner (interactive)",
                        role_name,
                    )
                    if role_name in self.role_registry._roles:
                        role_class = self.role_registry._roles[role_name]
                        role_instances[role_name] = (
                            self.showrunner.initialize_role_with_config(
                                role_class=role_class,
                                registry=self.provider_registry,
                                spec_path=self.spec_path,
                                human_callback=human_callback,
                                role_name=role_name,
                            )
                        )
                    else:
                        # This path is taken if a loop requires a role that is not
                        # registered in the role registry. This might happen if a
                        # role is defined in the spec but not implemented in code.
                        # We log a warning and continue, as the loop might be
                        # able to function without it.
                        logger.warning("Role '%s' not registered, skipping", role_name)

                else:
                    # Batch mode: Use existing registry logic for cached instances.
                    logger.trace(
                        "Getting role instance for '%s' from registry (batch)",
                        role_name,
                    )
                    role_instances[role_name] = self.role_registry.get_role(
                        role_name,
                        provider=self.provider,
                        provider_name=self.provider_name,
                    )

            except (KeyError, ValueError) as e:
                # This can happen if a role is not implemented, or if a provider
                # is not configured correctly. We log a warning and continue,
                # as the loop might be able to function without this role.
                logger.warning("Failed to instantiate role '%s': %s", role_name, e)
                pass

        logger.debug("Successfully instantiated %d roles", len(role_instances))

        # Create loop context
        project_info = self.workspace.get_project_info()
        logger.trace(
            "Creating LoopContext with %d artifacts and %d config items",
            len(artifacts or []),
            len(config or {}),
        )

        loop_context = LoopContext(
            loop_id=loop_id,
            project_id=project_id,
            workspace=self.workspace,
            role_instances=role_instances,
            artifacts=artifacts or [],
            project_metadata=project_info.model_dump(),
            config=config or {},
        )

        # Get loop class and instantiate
        logger.trace("Getting loop class for loop '%s'", loop_id)
        loop_class = self._get_loop_class(loop_id)
        loop_instance = loop_class(loop_context)
        logger.debug("Loop instance created, starting execution")

        # Execute loop
        logger.trace("Calling loop_instance.execute()")
        result = loop_instance.execute()

        if result.success:
            logger.info("Loop '%s' executed successfully", loop_id)
        else:
            logger.warning("Loop '%s' execution failed: %s", loop_id, result.error)

        return result

    def execute_goal(
        self,
        goal: str,
        project_id: str,
        project_state: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
        config: dict[str, Any] | None = None,
    ) -> LoopResult:
        """
        Execute workflow for a given goal (select loop + execute).

        Args:
            goal: User's goal or request
            project_id: Project identifier
            project_state: Current project state
            artifacts: Existing artifacts
            config: Loop configuration

        Returns:
            Loop execution result
        """
        logger.info("Executing goal workflow for project '%s': %s", project_id, goal)
        logger.debug(
            "Goal execution - artifacts=%d, config=%d",
            len(artifacts or []),
            len(config or {}),
        )

        # Select appropriate loop
        logger.trace("Calling select_loop()")
        loop_id = self.select_loop(goal, project_state, artifacts)

        # Execute the selected loop
        logger.trace("Calling execute_loop() with selected loop_id='%s'", loop_id)
        result = self.execute_loop(
            loop_id=loop_id,
            project_id=project_id,
            artifacts=artifacts,
            config=config,
        )

        if result.success:
            logger.info(
                "Goal workflow completed successfully for project '%s'", project_id
            )
        else:
            logger.warning(
                "Goal workflow failed for project '%s': %s", project_id, result.error
            )

        return result

    def _extract_loop_id(self, output: str) -> str:
        """
        Extract loop ID from showrunner output.

        For now, looks for common patterns like "Selected Loop: loop_id"
        or "**Selected Loop**: loop_id".

        Args:
            output: Showrunner output text

        Returns:
            Extracted loop ID

        Raises:
            RuntimeError: If loop ID cannot be extracted
        """
        logger.trace(
            "Attempting to extract loop ID from showrunner output (length=%d)",
            len(output),
        )

        # Look for patterns like "Selected Loop: story_spark"
        # or "**Selected Loop**: story_spark"
        patterns = [
            r"Selected Loop:?\s*[*]*\s*([a-z_]+)",
            r"\*\*Selected Loop\*\*:?\s*([a-z_]+)",
            r"Loop ID:?\s*[*]*\s*([a-z_]+)",
            r"loop_id:?\s*[`'\"]*([a-z_]+)[`'\"]*",
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                loop_id = match.group(1).lower()
                logger.trace(
                    "Pattern %d matched, extracted candidate loop_id='%s'", i, loop_id
                )
                # Verify it's a valid loop
                try:
                    self.loop_registry.get_loop_metadata(loop_id)
                    logger.debug("Extracted loop ID '%s' from pattern %d", loop_id, i)
                    return loop_id
                except KeyError:
                    logger.trace(
                        "Loop '%s' not found in registry, trying next pattern", loop_id
                    )
                    continue

        # Fallback: check if any loop_id appears in the output
        logger.trace("No pattern matches found, trying fallback lookup")
        for loop_metadata in self.loop_registry.list_loops():
            if loop_metadata.loop_id in output.lower():
                logger.debug(
                    "Found loop ID '%s' using fallback lookup", loop_metadata.loop_id
                )
                return loop_metadata.loop_id

        logger.error("Could not extract loop ID from showrunner output")
        raise RuntimeError(
            f"Could not extract loop ID from showrunner output:\n{output}"
        )

    def _get_loop_class(self, loop_id: str) -> type[Loop]:
        """
        Get loop class for the given loop ID.

        Args:
            loop_id: Loop identifier

        Returns:
            Loop class

        Raises:
            KeyError: If loop not implemented
        """
        logger.debug("Getting loop class for loop_id='%s'", loop_id)

        # Map loop IDs to loop classes
        # For now, only story_spark is implemented
        loop_classes = {
            "story_spark": self._import_story_spark_loop,
        }

        if loop_id not in loop_classes:
            logger.error(
                "Loop '%s' not yet implemented. Available: %s",
                loop_id,
                list(loop_classes.keys()),
            )
            raise KeyError(
                f"Loop '{loop_id}' not yet implemented. "
                f"Available loops: {list(loop_classes.keys())}"
            )

        logger.trace("Calling importer for loop '%s'", loop_id)
        loop_class = loop_classes[loop_id]()
        logger.debug(
            "Retrieved loop class %s for loop '%s'", loop_class.__name__, loop_id
        )
        return loop_class

    def _import_story_spark_loop(self) -> type[Loop]:
        """Import and return StorySparkLoop class."""
        from .loops.story_spark import StorySparkLoop

        return StorySparkLoop

    def __repr__(self) -> str:
        """String representation of orchestrator."""
        initialized = "initialized" if self.showrunner else "not initialized"
        return (
            f"Orchestrator({initialized}, "
            f"loops={len(self.loop_registry.list_loops())}, "
            f"roles={len(self.role_registry.list_roles())})"
        )
