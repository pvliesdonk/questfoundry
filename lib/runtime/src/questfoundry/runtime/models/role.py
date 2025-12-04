"""
Role profile models - represents a role_profile.schema.json definition.

Based on spec: components/node_factory.md
"""

from typing import Any


class RoleProfile:
    """Represents a role profile YAML definition."""

    def __init__(self, data: dict[str, Any]):
        """Initialize from parsed YAML data."""
        self.raw = data

        # Extract key fields
        self.id = data.get("id", "")
        identity = data.get("identity", {}) or {}
        self.name = identity.get("name", "")
        self.abbreviation = identity.get("abbreviation", "")
        self.role_type = data.get("role_type", "reasoning_agent")
        # Dormancy policy is defined under identity in compiled role profiles.
        # Fall back to top-level key for backward compatibility.
        self.dormancy_policy = identity.get(
            "dormancy_policy", data.get("dormancy_policy", "active")
        )

        # Behavior config
        behavior = data.get("behavior", {})
        self.prompt = behavior.get("prompt", {})
        self.tools = behavior.get("tools", [])
        self.model_config = behavior.get("model_config", {})

        # LLM config (check both execution.llm_config and root llm_config for compatibility)
        execution = data.get("execution", {})
        llm_config = execution.get("llm_config", data.get("llm_config", {}))

        if not self.model_config:
            self.model_config = {
                "provider": llm_config.get("provider", "auto"),
                "model_tier": llm_config.get(
                    "model_tier", "creative-writing"
                ),  # Changed: use valid tier
                "model": llm_config.get("model"),  # Backward compat (specific model name)
                "temperature": llm_config.get("temperature", 0.7),
                "max_tokens": llm_config.get("max_tokens", 4096),
            }

        # Protocol info
        protocol = data.get("protocol", {})
        self.can_send = protocol.get("can_send", [])
        self.can_receive = protocol.get("can_receive", [])

        # Wake condition (for default_dormant)
        self.wake_conditions = data.get("wake_conditions", [])

    def get_provider(self) -> str:
        """Get provider name from config (or 'auto' for automatic selection)."""
        return self.model_config.get("provider", "auto")

    def get_model(self) -> str:
        """Get model name from config (backward compatibility)."""
        return self.model_config.get("model", "claude-3-5-sonnet-20241022")

    def get_model_tier(self) -> str:
        """
        Get model tier from config.

        Returns:
            Model tier name (e.g., "creative-writing", "structured-thinking")
        """
        return self.model_config.get("model_tier", "creative-writing")

    def get_temperature(self) -> float:
        """Get temperature from config."""
        return float(self.model_config.get("temperature", 0.7))

    def get_max_tokens(self) -> int:
        """Get max_tokens from config."""
        return int(self.model_config.get("max_tokens", 4096))

    def should_execute(self, state: dict[str, Any]) -> bool:
        """
        Determine if this role should execute based on dormancy policy.

        Policies:
        - active: Always execute (default)
        - optional: Execute if explicitly requested in context
        - default_dormant: Execute only if wake condition is met
        """
        if self.dormancy_policy == "active":
            return True
        elif self.dormancy_policy == "optional":
            # Check if explicitly enabled in loop context
            return state.get("loop_context", {}).get(f"enable_{self.id}", False)
        elif self.dormancy_policy == "default_dormant":
            # Would need to evaluate wake_condition here
            # For now, return False (can be extended)
            return False
        return True
