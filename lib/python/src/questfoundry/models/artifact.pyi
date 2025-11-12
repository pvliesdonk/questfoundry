"""Type stubs for Artifact models"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ..validators import ValidationResult

class Artifact(BaseModel):
    """
    Base artifact model for QuestFoundry.

    Artifacts are the core data units in QuestFoundry, representing
    various types of creative content (hooks, scenes, canon, etc.).

    The artifact uses a flexible schema where:
    - `type` identifies the artifact type (e.g., 'hook_card')
    - `data` contains the artifact-specific content (validated against schema)
    - `metadata` contains common fields like id, timestamps, author
    """

    type: str
    data: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def artifact_id(self) -> str | None: ...
    @artifact_id.setter
    def artifact_id(self, value: str) -> None: ...
    @property
    def created(self) -> datetime | None: ...
    @created.setter
    def created(self, value: datetime) -> None: ...
    @property
    def modified(self) -> datetime | None: ...
    @modified.setter
    def modified(self, value: datetime) -> None: ...
    @property
    def author(self) -> str | None: ...
    @author.setter
    def author(self, value: str) -> None: ...
    def validate_schema(self) -> ValidationResult: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact: ...

class HookCard(Artifact):
    """Hook Card artifact.

    Small, traceable follow-up for new needs or uncertainties.
    Hooks are classified, routed to appropriate loops, and kept player-safe.
    """

    type: str

class TUBrief(Artifact):
    """Thematic Unit Brief artifact.

    Defines a unit of work to be performed in a loop.
    """

    type: str

class CanonPack(Artifact):
    """Canon Pack artifact.

    Authoritative lore and worldbuilding content.
    """

    type: str

class GatecheckReport(Artifact):
    """Gatecheck Report artifact.

    Quality validation report from the Gatekeeper role.
    """

    type: str

class CodexEntry(Artifact):
    """Codex Entry artifact.

    Player-facing encyclopedia entry.
    """

    type: str

class StyleAddendum(Artifact):
    """Style Addendum artifact.

    Style guide additions or modifications.
    """

    type: str

class ResearchMemo(Artifact):
    """Research Memo artifact.

    Research findings and references.
    """

    type: str

class Shotlist(Artifact):
    """Shotlist artifact.

    Visual composition and scene direction notes.
    """

    type: str

class Cuelist(Artifact):
    """Cuelist artifact.

    Audio cue timing and direction notes.
    """

    type: str

class ViewLog(Artifact):
    """View Log artifact.

    Record of view generation and export operations.
    """

    type: str

class ArtPlan(Artifact):
    """Art Plan artifact.

    Planning document for visual art direction.
    """

    type: str

class ArtManifest(Artifact):
    """Art Manifest artifact.

    Inventory of art assets.
    """

    type: str

class AudioPlan(Artifact):
    """Audio Plan artifact.

    Planning document for audio direction.
    """

    type: str

class EditNotes(Artifact):
    """Edit Notes artifact.

    Editorial feedback and revision notes.
    """

    type: str

class FrontMatter(Artifact):
    """Front Matter artifact.

    Book front matter content (title page, copyright, etc.).
    """

    type: str

class LanguagePack(Artifact):
    """Language Pack artifact.

    Translated content for a specific language.
    """

    type: str

class PNPlaytestNotes(Artifact):
    """PN Playtest Notes artifact.

    Player Narrator feedback from playtesting sessions.
    """

    type: str

class ProjectMetadata(Artifact):
    """Project Metadata artifact.

    Top-level project configuration and metadata.
    """

    type: str

class RegisterMap(Artifact):
    """Register Map artifact.

    Language register and tone mapping for characters/scenes.
    """

    type: str

class StyleManifest(Artifact):
    """Style Manifest artifact.

    Master style guide inventory.
    """

    type: str
