"""
Enums for runtime models.

These mirror the enum definitions in meta/schemas/core/_definitions.schema.json.
"""

from enum import Enum


class Archetype(str, Enum):
    """Agent archetype defining behavioral patterns."""

    ORCHESTRATOR = "orchestrator"
    CREATOR = "creator"
    VALIDATOR = "validator"
    RESEARCHER = "researcher"
    CURATOR = "curator"


class FieldType(str, Enum):
    """Data type for artifact fields."""

    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    URI = "uri"
    ARRAY = "array"
    OBJECT = "object"
    REF = "ref"


class StoreSemantics(str, Enum):
    """Storage behavior semantics."""

    APPEND_ONLY = "append_only"
    MUTABLE = "mutable"
    VERSIONED = "versioned"
    COLD = "cold"


class MessageType(str, Enum):
    """Type of inter-agent message."""

    DELEGATION_REQUEST = "delegation_request"
    DELEGATION_RESPONSE = "delegation_response"
    PROGRESS_UPDATE = "progress_update"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    NUDGE = "nudge"
    COMPLETION_SIGNAL = "completion_signal"
    LIFECYCLE_TRANSITION_REQUEST = "lifecycle_transition_request"
    LIFECYCLE_TRANSITION_RESPONSE = "lifecycle_transition_response"
    DIGEST = "digest"


class ArtifactCategory(str, Enum):
    """General category of artifact."""

    DOCUMENT = "document"
    RECORD = "record"
    MANIFEST = "manifest"
    COMPOSITE = "composite"
    DECISION = "decision"
    FEEDBACK = "feedback"


class AssetCategory(str, Enum):
    """General category of asset."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    DATA = "data"
    OTHER = "other"


class EnforcementType(str, Enum):
    """How a quality check is enforced."""

    RUNTIME = "runtime"
    LLM = "llm"


class BlockingBehavior(str, Enum):
    """Whether a check blocks progress or provides feedback only."""

    GATE = "gate"
    ADVISORY = "advisory"


class KnowledgeLayer(str, Enum):
    """Knowledge stratification layer determining access pattern."""

    CONSTITUTION = "constitution"
    MUST_KNOW = "must_know"
    SHOULD_KNOW = "should_know"
    ROLE_SPECIFIC = "role_specific"
    LOOKUP = "lookup"


class CompletionStatus(str, Enum):
    """Status of a completed delegation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ModelClass(str, Enum):
    """Model size class for tool and knowledge filtering.

    Used to optimize prompts for different model capacities:
    - SMALL: 8B parameters and under (local models)
    - MEDIUM: 9B-70B parameters
    - LARGE: 70B+ or cloud models (GPT-4, Claude)
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
