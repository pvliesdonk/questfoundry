"""Tool implementations for runtime internal operations."""

from questfoundry.runtime.tools.state_tools import (
    ReadColdSOT,
    ReadHotSOT,
    WriteColdSOT,
    WriteHotSOT,
)
from questfoundry.runtime.tools.protocol_tools import (
    SendProtocolEnvelope,
    SendProtocolMessage,
)
from questfoundry.runtime.tools.research_tools import WebSearch, LoreIndex
from questfoundry.runtime.tools.creative_tools import StableDiffusion
from questfoundry.runtime.tools.export_tools import (
    PandocConvert,
    PdfExport,
    ReadExports,
    WriteExports,
)
from questfoundry.runtime.tools.validation_tools import (
    EvaluateQualityBar,
    ValidateArtifact,
)
from questfoundry.runtime.tools.knowledge_tools import (
    ConsultGlossary,
    ConsultPlaybook,
    ConsultProtocol,
    ConsultQualityGate,
    ConsultRoleCharter,
)
from questfoundry.runtime.tools.orchestration_tools import (
    CreateSnapshot,
    SleepRole,
    TriggerGatecheck,
    UpdateTU,
    WakeRole,
)
from questfoundry.runtime.tools.media_tools import (
    GenerateAudio,
    GenerateImage,
)

__all__ = [
    # State tools
    "ReadHotSOT",
    "WriteHotSOT",
    "ReadColdSOT",
    "WriteColdSOT",
    # Protocol tools
    "SendProtocolMessage",
    "SendProtocolEnvelope",
    # Validation tools
    "ValidateArtifact",
    "EvaluateQualityBar",
    # Research tools
    "WebSearch",
    "LoreIndex",
    # Creative tools
    "StableDiffusion",
    # Export tools
    "PandocConvert",
    "PdfExport",
    "ReadExports",
    "WriteExports",
    # Knowledge tools (consult the cartridge)
    "ConsultPlaybook",
    "ConsultQualityGate",
    "ConsultProtocol",
    "ConsultRoleCharter",
    "ConsultGlossary",
    # Orchestration tools (Showrunner coordination)
    "CreateSnapshot",
    "UpdateTU",
    "WakeRole",
    "SleepRole",
    "TriggerGatecheck",
    # Media tools
    "GenerateImage",
    "GenerateAudio",
]
