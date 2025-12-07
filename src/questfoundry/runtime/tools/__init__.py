"""Runtime tools for SR and specialist roles."""

from questfoundry.runtime.tools.consult import (
    ConsultPlaybook,
    ConsultRoleCharter,
    ConsultSchema,
)
from questfoundry.runtime.tools.role import (
    ReadHotSot,
    ReturnToSR,
    WriteHotSot,
    read_hot_sot,
    write_hot_sot,
)
from questfoundry.runtime.tools.sr import (
    DelegateTo,
    ReadArtifact,
    Terminate,
    WriteArtifact,
)

__all__ = [
    # Consult tools (available to all roles)
    "ConsultPlaybook",
    "ConsultRoleCharter",
    "ConsultSchema",
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
]
