"""
Node Factory - transforms role profiles into LangGraph-compatible Runnable nodes.

Based on spec: components/node_factory.md
STRICT component - role → Runnable transformation is the core contract.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jinja2 import Template, TemplateError

from questfoundry.runtime.models.state import StudioState
from questfoundry.runtime.models.role import RoleProfile
from questfoundry.runtime.core.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class NodeFactory:
    """Transform role profiles into LangGraph-compatible Runnable nodes."""

    def __init__(self, schema_registry: Optional[SchemaRegistry] = None):
        """Initialize node factory."""
        self.schema_registry = schema_registry or SchemaRegistry()
        self._role_cache: Dict[str, RoleProfile] = {}

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

            # Resolve relative to spec directory
            if not Path(template_path).is_absolute():
                spec_root = Path(__file__).parent.parent.parent.parent.parent.parent / "spec"
                template_path = spec_root / "05-definitions" / template_path

            try:
                with open(template_path) as f:
                    template_str = f.read()
            except FileNotFoundError:
                logger.error(f"Template file not found: {template_path}")
                return f"[Template not found: {template_path}]"

        # Render template
        if template_engine == "jinja2":
            try:
                template = Template(template_str)

                # Prepare context
                context = {
                    "tu_id": state.get("tu_id", ""),
                    "tu_lifecycle": state.get("tu_lifecycle", ""),
                    "current_node": state.get("current_node", ""),
                    "loop_context": state.get("loop_context", {}),
                    "artifacts": state.get("artifacts", {}),
                    "quality_bars": state.get("quality_bars", {}),
                    "messages": state.get("messages", []),
                    "snapshot_ref": state.get("snapshot_ref"),
                    "error": state.get("error"),
                    "role_id": role.id,
                    "role_name": role.name
                }

                rendered = template.render(**context)
                return rendered

            except TemplateError as e:
                logger.error(f"Template rendering error in role {role.id}: {e}")
                raise

        else:
            logger.warning(f"Unknown template engine: {template_engine}")
            return template_str

    def select_llm(self, role: RoleProfile) -> Optional[Dict[str, Any]]:
        """
        Select appropriate LLM based on role type and model config.

        Role Types:
        1. reasoning_agent: Full LLM with complex reasoning
        2. production_executor: Thin LLM wrapper + heavy tool orchestration
        3. service: Pure tool execution, no LLM needed

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

        # Check for environment variable overrides
        provider = os.getenv("QF_LLM_PROVIDER", "anthropic")
        model = os.getenv("QF_DEFAULT_MODEL") or role.get_model()
        temperature = role.get_temperature()
        max_tokens = role.get_max_tokens()

        logger.info(f"Selected LLM provider: {provider}, model: {model}")

        return {
            "type": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "role_type": role.role_type
        }

    def create_role_node(self, role_id: str) -> Callable[[StudioState], StudioState]:
        """
        Create complete Runnable node for StateGraph.

        This is the main entry point used by GraphFactory.

        Returns a function with signature:
        def role_node(state: StudioState) -> StudioState:
            # Execute role logic
            # Update state
            # Return new state

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            Runnable node function compatible with LangGraph StateGraph
        """
        # Load role once
        role = self.load_role(role_id)

        def role_node(state: StudioState) -> StudioState:
            """Execute role and update state."""
            # 1. Check dormancy
            if not self.should_execute_role(role, state):
                logger.debug(f"Role {role.id} is dormant, skipping execution")
                return state

            # 2. Update current_node
            new_state = {**state}
            new_state["current_node"] = role.id

            try:
                # 3. Render prompt
                prompt = self.render_prompt(role, state)

                # 4. Invoke LLM or tools based on role type
                llm_config = self.select_llm(role)

                if llm_config:
                    provider = llm_config.get("type", "anthropic")

                    # Import appropriate LLM adapter
                    try:
                        if provider == "anthropic":
                            from questfoundry.runtime.plugins.llm.anthropic import (
                                AnthropicAdapter
                            )
                            adapter = AnthropicAdapter()
                        elif provider == "openai":
                            from questfoundry.runtime.plugins.llm.openai import (
                                OpenAIAdapter
                            )
                            adapter = OpenAIAdapter()
                        else:
                            raise ValueError(f"Unsupported LLM provider: {provider}")

                        # Get LLM instance
                        llm = adapter.get_llm(
                            model=llm_config["model"],
                            temperature=llm_config["temperature"],
                            max_tokens=llm_config["max_tokens"]
                        )

                        # Invoke LLM with prompt
                        logger.info(f"Invoking {provider} LLM for role {role.id}")
                        response = llm.invoke(prompt)
                        result = response.content if hasattr(response, 'content') else str(response)
                        logger.info(f"LLM response received: {len(result)} characters")

                    except Exception as e:
                        logger.error(f"LLM invocation failed for {role.id}: {e}")
                        raise
                else:
                    # Service type - tool-only execution (not yet implemented)
                    result = f"[{role.name}] Service execution (tools not yet implemented)"
                    logger.warning(f"Service-type role {role.id} executed without tools")

                # 5. Update state with artifact
                artifact = {
                    "artifact_type": "output",
                    "content": result,
                    "role_id": role.id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "tu_id": state["tu_id"],
                    "state_key": f"artifacts.hot.outputs.{role.id}",
                    "metadata": {
                        "prompt_length": len(prompt),
                        "model": role.get_model()
                    }
                }

                new_state["artifacts"] = {
                    **new_state.get("artifacts", {}),
                    role.id: artifact
                }

                # 6. Add protocol message
                message = {
                    "sender": role.id,
                    "receiver": "broadcast",
                    "intent": "artifact_created",
                    "payload": {
                        "artifact_id": role.id,
                        "artifact_type": "output"
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "envelope": {
                        "tu_id": state["tu_id"],
                        "snapshot_ref": state.get("snapshot_ref")
                    }
                }

                new_state["messages"] = new_state.get("messages", []) + [message]

                logger.info(f"Executed role: {role.id} ({role.name})")
                return new_state

            except Exception as e:
                logger.error(f"Error executing role {role.id}: {e}")
                new_state["error"] = str(e)
                new_state["retry_count"] = new_state.get("retry_count", 0) + 1
                return new_state

        return role_node
