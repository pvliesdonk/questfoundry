"""Runtime tools for SR and specialist roles."""

from questfoundry.runtime.tools.consult import (
    ConsultPlaybook,
    ConsultRoleCharter,
    ConsultSchema,
    ConsultTool,
)
from questfoundry.runtime.tools.registry import (
    UnavailableTool,
    build_agent_tools,
    register_tool,
)
from questfoundry.runtime.tools.role import (
    ReadHotSot,
    ReturnToSR,
    WriteHotSot,
    read_hot_sot,
    write_hot_sot,
)
from questfoundry.runtime.tools.playbook import (
    ConsultPlaybookV4,
    create_consult_playbook_tool,
)
from questfoundry.runtime.tools.searxng import (
    WebSearchTool,
    create_web_search_tool,
)
from questfoundry.runtime.tools.sr import (
    DelegateTo,
    ReadArtifact,
    Terminate,
    WriteArtifact,
)

__all__ = [
    # Registry (v4 capability-driven)
    "build_agent_tools",
    "register_tool",
    "UnavailableTool",
    # Consult tools (available to all roles)
    "ConsultPlaybook",
    "ConsultRoleCharter",
    "ConsultSchema",
    "ConsultTool",
    # SR tools
    "DelegateTo",
    "Terminate",
    "ReadArtifact",
    "WriteArtifact",
    # Role tools
    "ReturnToSR",
    "ReadHotSot",
    "WriteHotSot",
    "read_hot_sot",
    "write_hot_sot",
    # Lorekeeper tools
    "WebSearchTool",
    "create_web_search_tool",
    # Playbook tools (v4)
    "ConsultPlaybookV4",
    "create_consult_playbook_tool",
]
