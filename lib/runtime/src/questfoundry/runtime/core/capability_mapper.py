"""
CapabilityMapper - Maps abstract capabilities to concrete tool implementations.

This class bridges the gap between:
1. Abstract capability definitions in spec/05-definitions/capabilities.yaml
2. Concrete provider mappings in lib/runtime/config/tool_mappings.yaml
3. Runtime tool implementations in questfoundry.runtime.tools

Based on: Layer 5 role profiles (tools/capabilities) -> Layer 6 runtime tooling
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from questfoundry.runtime.core.schema_registry import DEFINITIONS_ROOT, SPEC_ROOT

logger = logging.getLogger(__name__)


class Provider:
    """Represents a single provider implementation for a capability."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize provider from configuration.

        Args:
            config: Provider configuration from tool_mappings.yaml
        """
        self.id: str = config["id"]
        self.type: str = config["type"]  # api_service, local_tool, stub
        self.tool_class: str = config["tool_class"]
        self.provider_name: str = config["provider_name"]
        self.availability_check: dict[str, Any] = config.get("availability_check", {})
        self.config: dict[str, Any] = config.get("config", {})
        self.fallback_strategy: str = config.get("fallback_strategy", "next_provider")
        self.priority: int = config.get("priority", 999)

    def is_available(self) -> bool:
        """
        Check if provider is available at runtime.

        Returns:
            True if provider is available, False otherwise
        """
        check_type = self.availability_check.get("type", "always_available")

        if check_type == "always_available":
            return True

        elif check_type == "api_key":
            # Check if API key environment variable is set
            env_var = self.availability_check.get("env_var")
            if not env_var:
                return False
            return bool(os.environ.get(env_var))

        elif check_type == "python_package":
            # Check if Python package is importable
            package_name = self.availability_check.get("package_name")
            if not package_name:
                return False
            try:
                __import__(package_name)
                return True
            except ImportError:
                return False

        elif check_type == "command_available":
            # Check if command is available in PATH
            command = self.availability_check.get("command")
            if not command:
                return False
            # Use shutil.which to check command availability
            import shutil
            return shutil.which(command) is not None

        else:
            logger.warning(f"Unknown availability check type: {check_type}")
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"Provider({self.id}, type={self.type}, priority={self.priority})"


class CapabilityMapper:
    """
    Maps abstract capabilities to concrete tool implementations.

    This class:
    - Loads capability definitions from capabilities.yaml
    - Loads provider mappings from tool_mappings.yaml
    - Selects providers based on availability and priority
    - Provides fallback chains when primary providers fail
    """

    def __init__(
        self,
        capabilities_path: str | Path | None = None,
        mappings_path: str | Path | None = None,
    ):
        """
        Initialize capability mapper.

        Args:
            capabilities_path: Path to capabilities.yaml (default: spec/05-definitions/)
            mappings_path: Path to tool_mappings.yaml (default: lib/runtime/config/)
        """
        # Default paths using same spec finding mechanism as SchemaRegistry
        if capabilities_path is None:
            if DEFINITIONS_ROOT:
                capabilities_path = DEFINITIONS_ROOT / "capabilities.yaml"
            else:
                # Fallback to relative path
                capabilities_path = Path("spec/05-definitions/capabilities.yaml")

        if mappings_path is None:
            # Tool mappings are in lib/runtime/config, not in spec
            # Try to find it relative to runtime module
            import questfoundry.runtime
            # questfoundry.runtime.__file__ is at lib/runtime/src/questfoundry/runtime/__init__.py
            # Go up 5 levels to reach monorepo root: runtime/ -> questfoundry/ -> src/ -> runtime/ -> lib/ -> root/
            monorepo_root = Path(questfoundry.runtime.__file__).parent.parent.parent.parent.parent
            mappings_path = monorepo_root / "lib" / "runtime" / "config" / "tool_mappings.yaml"

            # Fallback to relative path if that doesn't work
            if not mappings_path.exists():
                mappings_path = Path("lib/runtime/config/tool_mappings.yaml")

        self.capabilities_path = Path(capabilities_path)
        self.mappings_path = Path(mappings_path)

        # Loaded data
        self.capabilities: dict[str, Any] = {}
        self.mappings: dict[str, list[Provider]] = {}
        self.internal_tools: dict[str, Any] = {}
        self.knowledge_capabilities: dict[str, Any] = {}

        # Availability cache
        self._availability_cache: dict[str, bool] = {}

        # Load configurations
        self._load_capabilities()
        self._load_mappings()

    def _load_capabilities(self) -> None:
        """Load capability definitions from capabilities.yaml."""
        try:
            if not self.capabilities_path.exists():
                logger.warning(f"Capabilities file not found: {self.capabilities_path}")
                return

            with open(self.capabilities_path) as f:
                data = yaml.safe_load(f)

            # Store external capabilities by ID
            for cap in data.get("external_capabilities", []):
                cap_id = cap.get("id")
                if cap_id:
                    self.capabilities[cap_id] = cap

            # Store knowledge capabilities
            for cap in data.get("knowledge_capabilities", []):
                cap_id = cap.get("id")
                if cap_id:
                    self.knowledge_capabilities[cap_id] = cap

            logger.info(
                f"Loaded {len(self.capabilities)} external capabilities, "
                f"{len(self.knowledge_capabilities)} knowledge capabilities"
            )

        except Exception as e:
            logger.error(f"Failed to load capabilities: {e}")

    def _load_mappings(self) -> None:
        """Load tool mappings from tool_mappings.yaml."""
        try:
            if not self.mappings_path.exists():
                logger.warning(f"Tool mappings file not found: {self.mappings_path}")
                return

            with open(self.mappings_path) as f:
                data = yaml.safe_load(f)

            # Load external capability mappings
            external_mappings = data.get("external_capability_mappings", {})
            for cap_id, mapping_data in external_mappings.items():
                providers = []
                for provider_config in mapping_data.get("providers", []):
                    provider = Provider(provider_config)
                    providers.append(provider)

                # Sort by priority (lower number = higher priority)
                providers.sort(key=lambda p: p.priority)
                self.mappings[cap_id] = providers

            # Load knowledge capability mappings
            knowledge_mappings = data.get("knowledge_capability_mappings", {})
            for cap_id, mapping_data in knowledge_mappings.items():
                tool_class = mapping_data.get("tool_class")
                if tool_class:
                    # Create synthetic provider for knowledge capability
                    provider_config = {
                        "id": f"knowledge_{cap_id}",
                        "type": "knowledge",
                        "tool_class": tool_class,
                        "provider_name": mapping_data.get("description", "Knowledge Base"),
                        "availability_check": {"type": "always_available"},
                        "config": mapping_data.get("implementation", {}),
                        "priority": 1,
                    }
                    self.mappings[cap_id] = [Provider(provider_config)]

            # Store internal tools for reference (not exposed to agents)
            self.internal_tools = data.get("internal_tools", {})

            logger.info(f"Loaded mappings for {len(self.mappings)} capabilities")

        except Exception as e:
            logger.error(f"Failed to load tool mappings: {e}")

    def get_capability_info(self, capability_id: str) -> dict[str, Any] | None:
        """
        Get capability definition.

        Args:
            capability_id: Capability identifier (e.g., "image_generation")

        Returns:
            Capability definition or None if not found
        """
        if capability_id in self.capabilities:
            return self.capabilities[capability_id]
        if capability_id in self.knowledge_capabilities:
            return self.knowledge_capabilities[capability_id]
        return None

    def get_available_provider(
        self, capability_id: str, check_availability: bool = True
    ) -> Provider | None:
        """
        Get the best available provider for a capability.

        Args:
            capability_id: Capability identifier
            check_availability: Whether to check provider availability

        Returns:
            Provider instance or None if no provider available
        """
        if capability_id not in self.mappings:
            logger.warning(f"No mappings found for capability: {capability_id}")
            return None

        providers = self.mappings[capability_id]

        for provider in providers:
            # Check availability if requested
            if check_availability:
                cache_key = f"{capability_id}:{provider.id}"
                if cache_key in self._availability_cache:
                    is_available = self._availability_cache[cache_key]
                else:
                    is_available = provider.is_available()
                    self._availability_cache[cache_key] = is_available

                if not is_available:
                    logger.debug(
                        f"Provider {provider.id} not available for {capability_id}, trying next"
                    )
                    continue

            logger.info(f"Selected provider {provider.id} for capability {capability_id}")
            return provider

        logger.warning(f"No available providers for capability: {capability_id}")
        return None

    def get_tool_config_for_capability(
        self, capability_id: str, check_availability: bool = True
    ) -> dict[str, Any] | None:
        """
        Get tool configuration for a capability.

        Args:
            capability_id: Capability identifier
            check_availability: Whether to check provider availability

        Returns:
            Tool configuration dict with:
            - tool_class: Python class path for tool
            - config: Provider-specific configuration
            - provider_id: Selected provider ID
            - provider_name: Human-readable provider name
        """
        provider = self.get_available_provider(capability_id, check_availability)
        if not provider:
            return None

        return {
            "tool_class": provider.tool_class,
            "config": provider.config,
            "provider_id": provider.id,
            "provider_name": provider.provider_name,
            "capability_id": capability_id,
        }

    def get_all_providers(self, capability_id: str) -> list[Provider]:
        """
        Get all providers for a capability (including unavailable ones).

        Args:
            capability_id: Capability identifier

        Returns:
            List of Provider instances
        """
        return self.mappings.get(capability_id, [])

    def is_internal_tool(self, tool_id: str) -> bool:
        """
        Check if a tool is internal-only (not exposed to agents).

        Args:
            tool_id: Tool identifier

        Returns:
            True if tool is internal-only, False otherwise
        """
        for category_tools in self.internal_tools.values():
            if isinstance(category_tools, list):
                for tool in category_tools:
                    if tool.get("id") == tool_id:
                        return True
        return False

    def clear_availability_cache(self) -> None:
        """Clear the availability cache (force re-check on next query)."""
        self._availability_cache.clear()
        logger.debug("Cleared availability cache")

    def get_capability_summary(self) -> dict[str, Any]:
        """
        Get summary of all capabilities and their availability.

        Returns:
            Dictionary with capability availability status
        """
        summary = {
            "external_capabilities": {},
            "knowledge_capabilities": {},
            "total_capabilities": len(self.capabilities) + len(self.knowledge_capabilities),
        }

        # External capabilities
        for cap_id in self.capabilities:
            provider = self.get_available_provider(cap_id, check_availability=True)
            summary["external_capabilities"][cap_id] = {
                "available": provider is not None,
                "provider": provider.id if provider else None,
                "provider_name": provider.provider_name if provider else None,
            }

        # Knowledge capabilities (always available)
        for cap_id in self.knowledge_capabilities:
            summary["knowledge_capabilities"][cap_id] = {
                "available": True,
                "provider": "knowledge_base",
            }

        return summary
