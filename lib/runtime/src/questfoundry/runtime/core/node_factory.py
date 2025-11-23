"""
Node Factory - transforms role profiles into LangGraph-compatible Runnable nodes.

Based on spec: components/node_factory.md
STRICT component - role → Runnable transformation is the core contract.
"""

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files

from jinja2 import Template, TemplateError

from questfoundry.runtime.core.provider_manager import ProviderManager
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.models.role import RoleProfile
from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


class NodeFactory:
    """Transform role profiles into LangGraph-compatible Runnable nodes."""

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        state_manager: Any | None = None,
        preferred_provider: str | None = None,
    ):
        """Initialize node factory.

        Args:
            schema_registry: SchemaRegistry instance (creates new if not provided)
            state_manager: Optional StateManager for tracing messages
            preferred_provider: Preferred provider (e.g., "ollama") or fallback chain (e.g., "ollama,openai")
        """
        self.schema_registry = schema_registry or SchemaRegistry()
        self.provider_manager = ProviderManager()
        self._role_cache: dict[str, RoleProfile] = {}
        self.state_manager = state_manager  # For message tracing
        self.preferred_provider = preferred_provider  # Store for use in select_llm

    def load_role(self, role_id: str) -> RoleProfile:
        """
        Load and validate role YAML file.

        Steps:
        1. Construct path: spec/05-definitions/roles/{role_id}.yaml
        2. Load YAML with PyYAML
        3. Validate against role_profile.schema.json using SchemaRegistry
        4. Parse into structured RoleProfile object
        5. Return RoleProfile

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            RoleProfile object

        Raises:
            FileNotFoundError: If role YAML doesn't exist
            ValidationError: If YAML doesn't match schema
        """
        return self.schema_registry.load_role(role_id)

    def should_execute_role(self, role: RoleProfile, state: StudioState) -> bool:
        """
        Determine if role should execute based on dormancy policy.

        Dormancy Policies:
        - active: Always execute (default)
        - optional: Execute if explicitly requested in context
        - default_dormant: Execute only if wake_condition is met

        Args:
            role: RoleProfile object
            state: Current loop state

        Returns:
            True if role should execute, False if dormant
        """
        return role.should_execute(state)

    def render_prompt(self, role: RoleProfile, state: StudioState) -> str:
        """
        Render prompt template with state context using Jinja2.

        Steps:
        1. Load template (file:// or inline)
        2. Render with state context
        3. Return rendered prompt

        Args:
            role: RoleProfile with prompt template
            state: Current state for context

        Returns:
            Rendered prompt string

        Raises:
            FileNotFoundError: If template file doesn't exist
            TemplateError: If template rendering fails
        """
        prompt_config = role.prompt
        template_str = prompt_config.get("template", "")
        template_engine = prompt_config.get("template_engine", "jinja2")

        if not template_str:
            logger.warning(f"No prompt template for role {role.id}")
            return f"Execute role: {role.name}"

        # Load template
        if template_str.startswith("file://"):
            # File-based template
            template_path = template_str[7:]  # Remove "file://"
            template_str = None

            # Strategy 1: Try monorepo spec directory (development)
            if not Path(template_path).is_absolute():
                # Navigate up from lib/runtime/src/questfoundry/runtime/core/node_factory.py
                # to find spec/ directory at monorepo root
                # node_factory.py → core/ → runtime/ → questfoundry/ → src/ → runtime/ → lib/ → monorepo root
                monorepo_spec = (
                    Path(__file__).parent.parent.parent.parent.parent.parent.parent
                    / "spec"
                    / "05-definitions"
                    / template_path
                )
                if monorepo_spec.exists():
                    try:
                        template_str = monorepo_spec.read_text(encoding="utf-8")
                        logger.debug(f"Loaded template from monorepo: {monorepo_spec}")
                    except Exception as e:
                        logger.warning(f"Failed to read template from monorepo: {e}")

            # Strategy 2: Try downloaded spec (cached from GitHub releases)
            if template_str is None:
                try:
                    from questfoundry.runtime.core.spec_fetcher import get_cached_spec_path

                    cached_spec = get_cached_spec_path()
                    if cached_spec:
                        # Remove leading ../ from template path
                        clean_path = template_path.lstrip("../")
                        downloaded_template = cached_spec / "05-definitions" / clean_path
                        if downloaded_template.exists():
                            template_str = downloaded_template.read_text(encoding="utf-8")
                            logger.debug(f"Loaded template from downloaded spec: {clean_path}")
                except Exception as e:
                    logger.debug(f"Failed to load from downloaded spec: {e}")

            # Strategy 3: Try bundled resources (production)
            if template_str is None:
                try:
                    # Templates are bundled in resources/definitions/templates/
                    resource_path = template_path.lstrip("../")  # Remove leading ../
                    if resource_path.startswith("templates/"):
                        resource_path = resource_path[10:]  # Remove 'templates/' prefix

                    resource = files(
                        "questfoundry.runtime.resources.definitions.templates"
                    ).joinpath(resource_path)
                    template_str = resource.read_text(encoding="utf-8")
                    logger.debug(f"Loaded template from bundled resources: {resource_path}")
                except Exception as e:
                    logger.debug(f"Failed to load from bundled resources: {e}")

            # Strategy 4: Fallback - use system prompt only
            if template_str is None:
                logger.warning(
                    f"Template not found via any method, using fallback for role {role.id}"
                )
                # Return system prompt from role config if available
                system_prompt = role.model_config.get("system_prompt_prefix", "")
                if system_prompt:
                    return system_prompt
                return f"Execute role: {role.name}"

        # Render template
        if template_engine == "jinja2":
            try:
                template = Template(template_str)

                # Prepare context
                # Build tu_scope object from state
                tu_scope = {
                    "id": state.get("tu_id", ""),
                    "slice": state.get("loop_context", {}).get("slice", "full_work"),
                    "loop": state.get("loop_id", ""),
                    "deliverables": state.get("loop_context", {}).get("deliverables", []),
                    "timebox_minutes": state.get("loop_context", {}).get("timebox_minutes"),
                    "mode": state.get("loop_context", {}).get("mode", "workshop"),
                }

                # Build snapshot objects from state
                from datetime import datetime

                current_time = datetime.utcnow().isoformat() + "Z"

                current_snapshot = {
                    "id": state.get("snapshot_ref")
                    or f"SNAP-{state.get('tu_id', 'unknown')}-current",
                    "timestamp": state.get("updated_at", current_time),
                    "tu_id": state.get("tu_id", ""),
                    "lifecycle": state.get("tu_lifecycle", ""),
                }

                last_snapshot = {
                    "id": state.get("snapshot_ref") or f"SNAP-{state.get('tu_id', 'unknown')}-last",
                    "timestamp": state.get("created_at", current_time),
                    "tu_id": state.get("tu_id", ""),
                    "lifecycle": state.get("tu_lifecycle", ""),
                }

                # Extract hot_sot data for template variables
                hot_sot = state.get("hot_sot", {})
                cold_sot = state.get("cold_sot", {})

                context = {
                    "tu_id": state.get("tu_id", ""),
                    "tu_lifecycle": state.get("tu_lifecycle", ""),
                    "current_node": state.get("current_node", ""),
                    "loop_context": state.get("loop_context", {}),
                    "artifacts": state.get("artifacts", {}),  # Legacy, for debug
                    "hot_sot": hot_sot,  # Full hot SoT
                    "cold_sot": cold_sot,  # Full cold SoT
                    "quality_bars": state.get("quality_bars", {}),
                    "messages": state.get("messages", []),
                    "snapshot_ref": state.get("snapshot_ref"),
                    "error": state.get("error"),
                    "role_id": role.id,
                    "role_name": role.name,
                    # Template-expected variables from hot_sot
                    "section_briefs": hot_sot.get("section_briefs", []),
                    "style_addendum": hot_sot.get("style_notes"),
                    "lore_summaries": hot_sot.get("lore_notes"),
                    "codex_entries": cold_sot.get("codex", {}),
                    # Legacy variables
                    "tu_scope": tu_scope,
                    "current_snapshot": current_snapshot,
                    "last_snapshot": last_snapshot,
                    "current_tu": {  # For showrunner template
                        "id": state.get("tu_id", ""),
                        "slice": tu_scope["slice"],
                        "loop": tu_scope["loop"],
                        "status": state.get("tu_lifecycle", ""),
                        "deliverables": tu_scope["deliverables"],
                        "roles_awake": [],  # TODO: track active roles
                        "timebox_minutes": tu_scope.get("timebox_minutes"),
                        "risks": [],  # TODO: track risks
                    },
                    "customer_directive": state.get("loop_context", {}).get("customer_request", ""),
                }

                rendered = template.render(**context)
                return rendered

            except TemplateError as e:
                logger.error(f"Template rendering error in role {role.id}: {e}")
                raise

        else:
            logger.warning(f"Unknown template engine: {template_engine}")
            return template_str

    def extract_artifacts(self, role: RoleProfile, llm_output: str, tu_id: str) -> dict[str, Any]:
        """
        Extract artifacts from LLM JSON output and map to hot_sot keys.

        This bridges the gap between:
        1. LLM structured output fields (e.g., "briefs_written")
        2. Spec-defined hot_sot keys (e.g., "hot_sot.section_briefs")

        Args:
            role: RoleProfile with outputs specification
            llm_output: Raw JSON string from LLM
            tu_id: Trace Unit ID for artifact metadata

        Returns:
            Dict with hot_sot updates

        Example:
            Plotwright outputs {"briefs_written": ["id1", "id2"]}
            → Returns {"hot_sot": {"section_briefs": ["id1", "id2"]}}

        Note:
            Hot SoT is in-memory for Phase 1. Future: Redis/Memory backend.
            Cold SoT will use SQLite/File for persistence.
        """
        try:
            # 1. Parse JSON output
            output_data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON for {role.id}: {e}")
            logger.debug(f"Raw output: {llm_output[:500]}")
            return {}

        hot_sot_updates = {}

        # 2. Get role's output specification from YAML
        role_outputs = role.raw.get("interface", {}).get("outputs", [])

        # 3. Map LLM output fields to hot_sot keys
        # The structured_output schema defines what fields the LLM outputs
        # The outputs list defines where those should be stored (destination: hot → hot_sot)

        for output_spec in role_outputs:
            artifact_type = output_spec.get("artifact_type")
            state_key = output_spec.get("state_key", "")
            destination = output_spec.get("destination", "hot")

            if not state_key or destination != "hot":
                continue

            # Extract the hot_sot key from state_key
            # Examples:
            #   "artifacts.section_briefs" → "section_briefs" (legacy format, still works)
            #   "hot_sot.section_briefs" → "section_briefs"
            if state_key.startswith("artifacts."):
                hot_key = state_key.replace("artifacts.", "")
            elif state_key.startswith("hot_sot."):
                hot_key = state_key.replace("hot_sot.", "")
            else:
                hot_key = artifact_type

            # Try to find matching data in LLM output
            # Common patterns:
            # - section_briefs → briefs_written
            # - topology_notes → topology_updated / topology_notes
            # - hooks → hooks_proposed

            value = None

            # Try exact match first
            if hot_key in output_data:
                value = output_data[hot_key]
            # Try common field name patterns
            elif f"{artifact_type}_written" in output_data:
                value = output_data[f"{artifact_type}_written"]
            elif f"{artifact_type}s_written" in output_data:  # plural
                value = output_data[f"{artifact_type}s_written"]
            elif f"{artifact_type}_proposed" in output_data:
                value = output_data[f"{artifact_type}_proposed"]
            elif "briefs_written" in output_data and artifact_type == "section_briefs":
                # Specific mapping for plotwright briefs
                value = output_data["briefs_written"]

            if value is not None:
                hot_sot_updates[hot_key] = value
                logger.debug(f"Mapped {role.id} output to hot_sot.{hot_key}")

        # 4. Store full LLM output for debugging (in artifacts, not hot_sot)
        # This preserves raw output without polluting the hot_sot structure
        return {
            "hot_sot": hot_sot_updates,
            "artifacts": {f"_{role.id}_raw_output": output_data},  # Debug info
        }

    def select_llm(self, role: RoleProfile) -> dict[str, Any | None]:
        """
        Select appropriate LLM based on role type and model config.

        Role Types:
        1. reasoning_agent: Full LLM with complex reasoning
        2. production_executor: Thin LLM wrapper + heavy tool orchestration
        3. service: Pure tool execution, no LLM needed

        Uses ProviderManager for:
        - Provider auto-detection and selection
        - Model tier resolution (work-type → actual model)

        Args:
            role: RoleProfile with role_type and model_config

        Returns:
            LLM configuration dict or None for service type

        Note:
            Provider can be overridden via QF_LLM_PROVIDER environment variable.
            Model can be overridden via QF_DEFAULT_MODEL environment variable.
        """
        if role.role_type == "service":
            # Service type - no LLM needed
            return None

        # 1. Select provider (priority: env var > CLI > role YAML > auto)
        # Build provider preference with fallback chain if provided
        preferred_provider = (
            os.getenv("QF_LLM_PROVIDER") or self.preferred_provider or role.get_provider()
        )

        # If provider comes from env/CLI and is not "auto", treat it as
        # an explicit choice and enable strict provider selection. In that
        # mode we will not silently fall back to unrelated providers.
        env_or_cli_provider = os.getenv("QF_LLM_PROVIDER") or self.preferred_provider
        strict_provider_selection = bool(env_or_cli_provider) and preferred_provider != "auto"

        # Parse fallback chain if comma-separated (e.g., "ollama,openai")
        fallback_chain = None
        if preferred_provider and "," in preferred_provider:
            providers = [p.strip() for p in preferred_provider.split(",")]
            preferred_provider = providers[0]  # First is preferred
            fallback_chain = providers[1:]  # Rest are fallbacks

        provider = self.provider_manager.select_provider(
            preferred_provider, fallback_chain, strict=strict_provider_selection
        )

        # 2. Resolve model tier to actual model name
        # Check for explicit model override first (backward compatibility)
        if os.getenv("QF_DEFAULT_MODEL"):
            model = os.getenv("QF_DEFAULT_MODEL")
            logger.info(f"Using model override from QF_DEFAULT_MODEL: {model}")
        else:
            # Get tier from role (or use specific model if tier not specified)
            model_tier = role.get_model_tier()

            # Check if role specifies a specific model (backward compatibility)
            specific_model = role.model_config.get("model")
            if specific_model:
                # Use specific model directly
                model = specific_model
                logger.info(f"Using specific model from role config: {model}")
            else:
                # Resolve tier to provider-specific model
                model = self.provider_manager.resolve_model(provider, model_tier)
                logger.info(f"Resolved tier '{model_tier}' to {provider}:{model}")

        # 3. Get other parameters
        temperature = role.get_temperature()
        max_tokens = role.get_max_tokens()

        logger.info(
            f"Selected LLM - provider: {provider}, model: {model}, temperature: {temperature}"
        )

        return {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "role_type": role.role_type,
            # Provider selection metadata used for graceful fallback
            "fallback_chain": fallback_chain,
            "strict_provider_selection": strict_provider_selection,
        }

    def create_role_node(self, role_id: str) -> Callable[[StudioState], dict[str, Any]]:
        """
        Create complete Runnable node for StateGraph.

        This is the main entry point used by GraphFactory.

        Returns a function with signature:
        def role_node(state: StudioState) -> dict[str, Any]:
            # Execute role logic
            # Return partial state update (only changed fields)

        This follows LangGraph's recommended pattern for avoiding concurrent
        update conflicts: nodes return only the fields they want to change,
        not the entire state dict.

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            Runnable node function compatible with LangGraph StateGraph
        """
        # Load role once
        role = self.load_role(role_id)

        def role_node(state: StudioState) -> dict[str, Any]:
            """Execute role and update state.

            Returns:
                Partial state update dict (only changed fields)
            """
            # 1. Check dormancy
            if not self.should_execute_role(role, state):
                logger.debug(f"Role {role.id} is dormant, skipping execution")
                return {}  # Return empty dict = no state changes

            try:
                # 2. Send "role_started" progress indicator (live feedback)
                if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                    if self.state_manager._trace_handler:
                        try:
                            start_message = {
                                "sender": role.id,
                                "receiver": "system",
                                "intent": "role_started",
                                "payload": {
                                    "role_name": role.name,
                                },
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "envelope": {
                                    "tu_id": state["tu_id"],
                                },
                            }
                            self.state_manager._trace_handler.trace_message(start_message)
                        except Exception as e:
                            logger.warning(f"Trace handler error: {e}")

                # 3. Render prompt
                prompt = self.render_prompt(role, state)

                # 4. Invoke LLM or tools based on role type
                llm_config = self.select_llm(role)

                if llm_config:
                    provider = llm_config.get("provider")

                    # Create LLM client via ProviderManager (cached)
                    try:
                        llm = self.provider_manager.create_llm_client(
                            provider=provider,
                            model=llm_config["model"],
                            temperature=llm_config["temperature"],
                            max_tokens=llm_config["max_tokens"],
                        )

                        # Add JSON format instructions for all providers
                        # Critical: All roles MUST output valid JSON
                        json_prompt = (
                            f"{prompt}\n\n"
                            "IMPORTANT: Your response MUST be valid JSON only. "
                            "Do not include any text before or after the JSON object. "
                            "Do not use markdown code blocks. "
                            "Output only the raw JSON object."
                        )

                        # Invoke LLM with JSON-instructed prompt
                        logger.info(f"Invoking {provider} LLM for role {role.id}")
                        response = llm.invoke(json_prompt)
                        result = response.content if hasattr(response, "content") else str(response)

                        # Send role_completed with extracted insight and prompt context
                        if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                            if self.state_manager._trace_handler:
                                try:
                                    # Include FULL result for trace (no truncation)
                                    # Critical: Roles need to see complete messages to self-stabilize
                                    # Also include the rendered prompt so we can debug
                                    # cross-role communication and prompt construction.
                                    completed_message = {
                                        "sender": role.id,
                                        "receiver": "system",
                                        "intent": "role_completed",
                                        "payload": {
                                            "role_name": role.name,
                                            "insight": result,  # Full result, not truncated
                                            "length": len(result),
                                            # NOTE: This is the rendered role prompt *before*
                                            # JSON-format instructions are appended.
                                            # TODO: Ideally we'd also capture a structured view
                                            # of the prompt components (system vs. user vs. tools)
                                            # rather than a single flattened string.
                                            "prompt": prompt,
                                        },
                                        "timestamp": datetime.utcnow().isoformat() + "Z",
                                        "envelope": {
                                            "tu_id": state["tu_id"],
                                        },
                                    }
                                    self.state_manager._trace_handler.trace_message(
                                        completed_message
                                    )
                                except Exception as e:
                                    logger.warning(f"Trace handler error: {e}")

                    except Exception as e:
                        # Gracefully handle provider-level failures (e.g. 404, rate limit)
                        logger.error(f"LLM invocation failed for {role.id}: {e}")
                        # Mark provider as unavailable for subsequent selections in this run
                        if provider:
                            self.provider_manager.mark_unavailable(
                                provider, reason=str(e)
                            )
                        # Re-raise so the caller can decide whether to abort or retry
                        raise
                else:
                    # Service type - tool-only execution (not yet implemented)
                    result = f"[{role.name}] Service execution (tools not yet implemented)"
                    logger.warning(f"Service-type role {role.id} executed without tools")

                # 4. Extract artifacts from LLM output
                extracted_artifacts = self.extract_artifacts(role, result, state["tu_id"])

                # 5. Create protocol message
                message = {
                    "sender": role.id,
                    "receiver": "broadcast",
                    "intent": "artifact_created",
                    "payload": {"artifact_id": role.id, "artifact_type": "output"},
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "envelope": {
                        "tu_id": state["tu_id"],
                        "snapshot_ref": state.get("snapshot_ref"),
                    },
                }

                # 6. Trace message if state_manager has trace handler
                if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                    if self.state_manager._trace_handler:
                        try:
                            self.state_manager._trace_handler.trace_message(message)
                        except Exception as e:
                            logger.warning(f"Trace handler error: {e}")

                # 7. Return extracted artifacts mapped to spec-defined state keys
                # Note: Reducers (Annotated types) handle merging artifacts and messages
                # Don't manually merge - just return the new values
                # Unpack extracted_artifacts to put hot_sot and artifacts at top level
                return {
                    **extracted_artifacts,  # Unpacks hot_sot and artifacts
                    "messages": [message],
                }

            except Exception as e:
                logger.error(f"Error executing role {role.id}: {e}")
                # Return partial update with error info
                return {"error": str(e), "retry_count": state.get("retry_count", 0) + 1}

        return role_node

    def create_multi_role_node(
        self, sub_nodes: list[dict[str, Any]], node_id: str
    ) -> Callable[[StudioState], dict[str, Any]]:
        """
        Create a multi-role parallel execution node.

        For nodes with role="Multi" and parallel_execution=true, this creates
        a wrapper that executes multiple roles in parallel and merges their results.

        Args:
            sub_nodes: List of sub-node definitions, each with 'role' and 'task'
            node_id: Node identifier for logging

        Returns:
            Runnable node function that executes roles in parallel

        Example sub_nodes structure:
            [
                {"role": "Lore Weaver", "task": "Review narrative hooks..."},
                {"role": "Plotwright", "task": "Review structural impact..."}
            ]
        """
        import concurrent.futures

        # Load all role nodes upfront
        role_execution_pairs = []
        for sub_node in sub_nodes:
            role_name = sub_node.get("role", "")
            task = sub_node.get("task", "")

            # Convert role name to role_id (lowercase, replace spaces with underscores)
            role_id = role_name.lower().replace(" ", "_")

            try:
                # Create role node for this sub-role
                role_node = self.create_role_node(role_id)
                role_execution_pairs.append((role_name, role_id, role_node, task))
                logger.debug(f"Loaded role for parallel execution: {role_name} ({role_id})")
            except Exception as e:
                logger.warning(
                    f"Failed to load role {role_name} ({role_id}) for parallel node: {e}"
                )
                # Continue with other roles even if one fails to load

        if not role_execution_pairs:
            logger.error(f"No valid roles loaded for multi-role node {node_id}")

            # Return a placeholder that logs the error
            def error_node(state: StudioState) -> dict[str, Any]:
                return {"error": f"Multi-role node {node_id} has no valid roles"}

            return error_node

        def multi_role_node(state: StudioState) -> dict[str, Any]:
            """Execute multiple roles in parallel and merge results."""
            logger.info(
                f"[bold cyan]Executing parallel node:[/bold cyan] {node_id} with {len(role_execution_pairs)} roles"
            )

            # Execute all roles in parallel using ThreadPoolExecutor
            results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(role_execution_pairs)
            ) as executor:
                # Submit all role executions
                futures = {}
                for role_name, role_id, role_node, task in role_execution_pairs:
                    logger.debug(f"Submitting parallel task for {role_name}: {task[:80]}...")
                    future = executor.submit(role_node, state)
                    futures[future] = (role_name, role_id)

                # Collect results as they complete
                for future in concurrent.futures.as_completed(
                    futures, timeout=600
                ):  # 10 min total timeout
                    role_name, role_id = futures[future]
                    try:
                        result = future.result(timeout=300)  # 5 min per role
                        logger.info(f"[green]✓[/green] {role_name} completed in parallel")
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Role {role_name} timed out in parallel execution")
                        results.append({"error": f"Role {role_name} timed out"})
                    except Exception as e:
                        logger.error(f"Role {role_name} failed in parallel execution: {e}")
                        results.append({"error": f"Role {role_name} failed: {str(e)}"})

            # Merge all results into single state update
            merged_artifacts = {}
            merged_messages = []
            errors = []

            for result in results:
                # Merge artifacts
                if "artifacts" in result and isinstance(result["artifacts"], dict):
                    merged_artifacts.update(result["artifacts"])

                # Merge messages
                if "messages" in result and isinstance(result["messages"], list):
                    merged_messages.extend(result["messages"])

                # Collect errors
                if "error" in result:
                    errors.append(result["error"])

            # Build final merged state update
            merged_state = {"artifacts": merged_artifacts, "messages": merged_messages}

            # If any errors occurred, include them in the merged state
            if errors:
                merged_state["error"] = "; ".join(errors)
                logger.warning(
                    f"Parallel node {node_id} completed with errors: {merged_state['error']}"
                )
            else:
                logger.info(
                    f"[bold green]Parallel node {node_id} completed successfully[/bold green]"
                )

            return merged_state

        return multi_role_node
