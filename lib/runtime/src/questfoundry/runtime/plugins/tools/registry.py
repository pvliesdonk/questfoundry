"""
Tool Registry - provides tool implementations for roles.

Based on spec: interfaces/tool_registry.yaml
Implements plugin provider pattern for tool access.
"""

import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from questfoundry.runtime.tools import (
    # Knowledge tools
    ConsultGlossary,
    ConsultPlaybook,
    ConsultProtocol,
    ConsultQualityGate,
    ConsultRoleCharter,
    # Orchestration tools
    CreateSnapshot,
    # Validation tools
    EvaluateQualityBar,
    # Media tools
    GenerateAudio,
    # Research tools
    LoreIndex,
    # Export tools
    PandocConvert,
    PdfExport,
    # State tools
    ReadColdSOT,
    ReadExports,
    ReadHotSOT,
    # Protocol tools
    SendProtocolEnvelope,
    SendProtocolMessage,
    SleepRole,
    # Creative tools
    StableDiffusion,
    TriggerGatecheck,
    UpdateTU,
    ValidateArtifact,
    WakeRole,
    WebSearch,
    WriteColdSOT,
    WriteExports,
    WriteHotSOT,
)

logger = logging.getLogger(__name__)

# Global strict mode for LangChain BaseTool to satisfy OpenAI function-calling
try:
    BaseTool.strict = True
except Exception:
    pass


class Tool:
    """Base class for tools."""

    def __init__(self, tool_id: str, name: str, description: str):
        """Initialize tool."""
        self.tool_id = tool_id
        self.name = name
        self.description = description

    def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool (override in subclasses)."""
        raise NotImplementedError(f"Tool {self.tool_id} not implemented")


class LangChainToolAdapter(Tool):
    """Adapter to wrap LangChain BaseTool instances in the registry API."""

    def __init__(self, tool_id: str, base_tool: Any):
        super().__init__(
            tool_id, getattr(base_tool, "name", tool_id), getattr(base_tool, "description", "")
        )
        # Ensure compatibility with OpenAI function tools: require strict schema
        try:
            base_tool.strict = True  # Needed for OpenAI function tool parsing
        except Exception:
            pass
        self._base_tool = base_tool

    def to_langchain_tool(self) -> Any:
        """Expose the underlying BaseTool with strict schema intact."""
        return self._base_tool

    def invoke(self, **kwargs: Any) -> Any:
        logger.info(
            f"[TOOL] {self.tool_id} invoked", extra={"tool_id": self.tool_id, "args": kwargs}
        )
        # Call _run() directly - BaseTool.run() expects positional tool_input,
        # but _run() takes **kwargs which matches our invocation pattern
        return self._base_tool._run(**kwargs)


class MockTool(Tool):
    """Mock tool for testing - returns dummy data."""

    def __init__(self, tool_id: str, name: str, description: str):
        """Initialize mock tool."""
        super().__init__(tool_id, name, description)

    def invoke(self, **kwargs: Any) -> Any:
        """Return mock response."""
        logger.info(
            f"[TOOL] {self.tool_id} invoked (mock)", extra={"tool_id": self.tool_id, "args": kwargs}
        )
        return {
            "status": "success",
            "tool": self.tool_id,
            "message": f"Mock execution of {self.name}",
            "input": kwargs,
        }


class ToolRegistry:
    """
    Registry for tool implementations.

    Provides tool lookup and registration.
    """

    def __init__(self):
        """Initialize registry with stub tools."""
        self._tools: dict[str, Tool] = {}
        self._register_stub_tools()

    def _register_stub_tools(self) -> None:
        """Register stub implementations for common tools."""
        # Image generation (provider-aware)
        self._tools["stable_diffusion"] = LangChainToolAdapter(
            "stable_diffusion", StableDiffusion()
        )

        # Audio synthesis (kept mock until a backend is configured)
        self._tools["audio_synthesis"] = MockTool(
            "audio_synthesis", "Audio Synthesis", "Generate audio/music using synthesis"
        )

        # Document conversion / export
        self._tools["pandoc"] = LangChainToolAdapter("pandoc", PandocConvert())
        self._tools["pdf_export"] = LangChainToolAdapter("pdf_export", PdfExport())
        self._tools["epub_export"] = LangChainToolAdapter(
            "epub_export", PdfExport(output_format="epub")
        )

        # Web search
        self._tools["web_search"] = LangChainToolAdapter("web_search", WebSearch())

        # Lore index lookup
        self._tools["lore_index"] = LangChainToolAdapter("lore_index", LoreIndex())

        # Internal state/protocol/validation tools
        self._tools["read_hot_sot"] = LangChainToolAdapter("read_hot_sot", ReadHotSOT())
        self._tools["write_hot_sot"] = LangChainToolAdapter("write_hot_sot", WriteHotSOT())
        self._tools["read_cold_sot"] = LangChainToolAdapter("read_cold_sot", ReadColdSOT())
        self._tools["write_cold_sot"] = LangChainToolAdapter("write_cold_sot", WriteColdSOT())
        self._tools["send_protocol_message"] = LangChainToolAdapter(
            "send_protocol_message", SendProtocolMessage()
        )
        self._tools["send_protocol_envelope"] = LangChainToolAdapter(
            "send_protocol_envelope", SendProtocolEnvelope()
        )
        self._tools["validate_artifact"] = LangChainToolAdapter(
            "validate_artifact", ValidateArtifact()
        )
        self._tools["evaluate_quality_bar"] = LangChainToolAdapter(
            "evaluate_quality_bar", EvaluateQualityBar()
        )

        # Knowledge tools (consult the cartridge/spec)
        self._tools["consult_playbook"] = LangChainToolAdapter(
            "consult_playbook", ConsultPlaybook()
        )
        self._tools["consult_quality_gate"] = LangChainToolAdapter(
            "consult_quality_gate", ConsultQualityGate()
        )
        self._tools["consult_protocol"] = LangChainToolAdapter(
            "consult_protocol", ConsultProtocol()
        )
        self._tools["consult_role_charter"] = LangChainToolAdapter(
            "consult_role_charter", ConsultRoleCharter()
        )
        self._tools["consult_glossary"] = LangChainToolAdapter(
            "consult_glossary", ConsultGlossary()
        )

        # Orchestration tools (Showrunner coordination)
        self._tools["create_snapshot"] = LangChainToolAdapter(
            "create_snapshot", CreateSnapshot()
        )
        self._tools["update_tu"] = LangChainToolAdapter("update_tu", UpdateTU())
        self._tools["wake_role"] = LangChainToolAdapter("wake_role", WakeRole())
        self._tools["sleep_role"] = LangChainToolAdapter("sleep_role", SleepRole())
        self._tools["trigger_gatecheck"] = LangChainToolAdapter(
            "trigger_gatecheck", TriggerGatecheck()
        )

        # Media tools - aliases to actual implementations
        # generate_image -> StableDiffusion (already implements multi-provider image gen)
        self._tools["generate_image"] = LangChainToolAdapter(
            "generate_image", StableDiffusion()
        )
        # generate_audio -> stub (no audio provider implemented yet)
        self._tools["generate_audio"] = LangChainToolAdapter(
            "generate_audio", GenerateAudio()
        )

        # Export tools (read/write exports)
        self._tools["read_exports"] = LangChainToolAdapter(
            "read_exports", ReadExports()
        )
        self._tools["write_exports"] = LangChainToolAdapter(
            "write_exports", WriteExports()
        )

        logger.info(f"Registered {len(self._tools)} stub tools")

    def get_tool(self, tool_id: str, config: dict[str, Any | None] = None) -> Tool | None:
        """
        Get tool by ID.

        Args:
            tool_id: Tool identifier
            config: Optional tool configuration

        Returns:
            Tool instance or None if not found
        """
        if tool_id not in self._tools:
            logger.warning(f"Tool not found: {tool_id}")
            return None

        tool = self._tools[tool_id]

        # Store config if provided (can be used by tool implementations)
        if config:
            tool.config = config

        return tool

    def register_tool(self, tool_id: str, tool: Tool) -> None:
        """
        Register a tool implementation.

        Args:
            tool_id: Tool identifier
            tool: Tool instance
        """
        self._tools[tool_id] = tool
        logger.info(f"Registered tool: {tool_id}")

    def list_available_tools(self) -> list[str]:
        """
        List all available tools.

        Returns:
            List of tool IDs
        """
        return list(self._tools.keys())

    def has_tool(self, tool_id: str) -> bool:
        """
        Check if tool is available.

        Args:
            tool_id: Tool identifier

        Returns:
            True if tool exists, False otherwise
        """
        return tool_id in self._tools

    def get_all_tools(self) -> dict[str, Tool]:
        """
        Get all tools.

        Returns:
            Dict of all tools
        """
        return self._tools.copy()


# Global registry instance
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """
    Get or create tool registry singleton.

    Returns:
        ToolRegistry instance
    """
    global _registry

    if _registry is None:
        _registry = ToolRegistry()

    return _registry


def get_tool(tool_id: str, config: dict[str, Any | None] = None) -> Tool | None:
    """
    Convenience function to get tool from registry.

    Args:
        tool_id: Tool identifier
        config: Optional tool configuration

    Returns:
        Tool instance or None if not found
    """
    registry = get_tool_registry()
    return registry.get_tool(tool_id, config)
