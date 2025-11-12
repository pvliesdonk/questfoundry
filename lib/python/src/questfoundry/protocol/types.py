"""Type definitions for protocol models"""

from enum import Enum


class HotCold(str, Enum):
    """Hot/Cold workspace designation"""

    HOT = "hot"
    COLD = "cold"


class SpoilerPolicy(str, Enum):
    """Spoiler content policy"""

    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"


class RoleName(str, Enum):
    """QuestFoundry role names (Layer 5)"""

    # Core roles
    SHOWRUNNER = "SR"
    GATEKEEPER = "GK"
    PLOTWRIGHT = "PW"
    SCENE_SMITH = "SS"
    STYLE_LEAD = "ST"
    LORE_WEAVER = "LW"
    CODEX_CURATOR = "CC"
    ART_DIRECTOR = "AD"
    ILLUSTRATOR = "IL"
    AUDIO_DIRECTOR = "AuD"
    AUDIO_PRODUCER = "AuP"
    TRANSLATOR = "TR"
    BOOK_BINDER = "BB"
    PLAYER_NARRATOR = "PN"
    RESEARCHER = "RS"


class Intent(str, Enum):
    """
    Protocol intent verbs (Layer 4).

    Intents use hierarchical dot notation: category.action
    New intents can be added as strings following the pattern: ^[a-z]+([._-][a-z]+)*$
    """

    # Scene intents
    SCENE_WRITE = "scene.write"
    SCENE_EDIT = "scene.edit"
    SCENE_REVIEW = "scene.review"

    # Hook intents
    HOOK_CLASSIFY = "hook.classify"
    HOOK_HARVEST = "hook.harvest"

    # Quality intents
    QUALITY_CHECK = "quality.check"
    QUALITY_VALIDATE = "quality.validate"

    # Canon workflow intents (Layer 6/7)
    CANON_TRANSFER_EXPORT = "canon.transfer.export"
    CANON_TRANSFER_IMPORT = "canon.transfer.import"
    CANON_GENESIS_CREATE = "canon.genesis.create"

    # Lore intents
    LORE_DEEPEN = "lore.deepen"
    LORE_RESEARCH = "lore.research"

    # Codex intents
    CODEX_UPDATE = "codex.update"
    CODEX_EXPAND = "codex.expand"

    # Art intents
    ART_PLAN = "art.plan"
    ART_GENERATE = "art.generate"

    # Audio intents
    AUDIO_PLAN = "audio.plan"
    AUDIO_GENERATE = "audio.generate"

    # Style intents
    STYLE_TUNE = "style.tune"
    STYLE_VALIDATE = "style.validate"

    # Binding intents
    BINDING_RUN = "binding.run"
    BINDING_EXPORT = "binding.export"


# Intent constants for canon workflows (Layer 6/7 specification)
CANON_TRANSFER_EXPORT = "canon.transfer.export"
CANON_TRANSFER_IMPORT = "canon.transfer.import"
CANON_GENESIS_CREATE = "canon.genesis.create"
