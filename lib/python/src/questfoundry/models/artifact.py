"""Pydantic models for QuestFoundry artifacts"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..validators import ValidationResult, validate_artifact


class Artifact(BaseModel):
    """
    Base artifact model for QuestFoundry.

    Artifacts are the core data units in QuestFoundry, representing
    various types of creative content (hooks, scenes, canon, etc.).
    Each artifact flows through QuestFoundry's hot/cold workflow:
    - Hot: Work-in-progress artifacts stored as individual files
    - Cold: Ship-ready artifacts validated and stored in SQLite

    Structure:
        - `type`: Identifies the artifact type (e.g., 'hook_card', 'canon_pack')
        - `data`: Artifact-specific content validated against JSON schema
        - `metadata`: Common fields (id, timestamps, author, temperature, etc.)

    The flexible schema design allows each artifact type to have different
    required fields while sharing common metadata and validation infrastructure.

    Typical workflow:
        1. Create artifact in hot workspace
        2. Validate against schema using validate_schema()
        3. Pass through quality bars via Gatekeeper
        4. Promote to cold storage when ship-ready
        5. Export to player-safe views

    Examples:
        Create and validate a hook card:
            >>> artifact = Artifact(
            ...     type="hook_card",
            ...     data={
            ...         "header": {"short_name": "Dragon Rumor"},
            ...         "summary": "Merchant mentions dragon sighting"
            ...     },
            ...     metadata={"id": "HOOK-001", "author": "Writer"}
            ... )
            >>> result = artifact.validate_schema()
            >>> if result.is_valid:
            ...     print("Artifact is valid!")

        Access metadata properties:
            >>> artifact.artifact_id = "HOOK-001"  # Set ID
            >>> print(artifact.artifact_id)  # Get ID: "HOOK-001"
            >>> artifact.author = "Writer"  # Set author
            >>> print(artifact.author)  # Get author: "Writer"
            >>> print(artifact.created)  # Get created timestamp (or None)

        Serialize for storage:
            >>> data = artifact.to_dict()
            >>> loaded = Artifact.from_dict(data)
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"type": "hook_card", "data": {}, "metadata": {}}}
    )

    type: str = Field(..., description="Artifact type (e.g., 'hook_card')")
    data: dict[str, Any] = Field(default_factory=dict, description="Artifact data")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Artifact metadata"
    )

    # Common metadata accessors

    @property
    def artifact_id(self) -> str | None:
        """Get artifact ID from metadata"""
        return self.metadata.get("id")

    @artifact_id.setter
    def artifact_id(self, value: str) -> None:
        """Set artifact ID in metadata"""
        self.metadata["id"] = value

    @property
    def created(self) -> datetime | None:
        """
        Get creation timestamp from metadata.

        Returns:
            Datetime object if valid timestamp exists, None otherwise
        """
        created = self.metadata.get("created")
        if isinstance(created, str):
            try:
                return datetime.fromisoformat(created)
            except ValueError:
                # Invalid ISO format - return None rather than raise
                return None
        if isinstance(created, datetime):
            return created
        return None

    @created.setter
    def created(self, value: datetime) -> None:
        """Set creation timestamp in metadata"""
        self.metadata["created"] = value.isoformat()

    @property
    def modified(self) -> datetime | None:
        """
        Get modification timestamp from metadata.

        Returns:
            Datetime object if valid timestamp exists, None otherwise
        """
        modified = self.metadata.get("modified")
        if isinstance(modified, str):
            try:
                return datetime.fromisoformat(modified)
            except ValueError:
                # Invalid ISO format - return None rather than raise
                return None
        if isinstance(modified, datetime):
            return modified
        return None

    @modified.setter
    def modified(self, value: datetime) -> None:
        """Set modification timestamp in metadata"""
        self.metadata["modified"] = value.isoformat()

    @property
    def author(self) -> str | None:
        """Get author from metadata"""
        author = self.metadata.get("author")
        if isinstance(author, str):
            return author
        return None

    @author.setter
    def author(self, value: str) -> None:
        """Set author in metadata"""
        self.metadata["author"] = value

    # Validation methods

    def validate_schema(self) -> ValidationResult:
        """
        Validate artifact against its schema.

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        instance = {"type": self.type, "data": self.data, "metadata": self.metadata}
        return validate_artifact(instance, self.type)

    # Serialization methods

    def to_dict(self) -> dict[str, Any]:
        """
        Convert artifact to dictionary.

        Returns:
            Dictionary representation of the artifact
        """
        return {
            "type": self.type,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        """
        Create artifact from dictionary.

        Args:
            data: Dictionary with type, data, and metadata fields

        Returns:
            New Artifact instance

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        return cls.model_validate(data)


class HookCard(Artifact):
    """
    Hook Card artifact.

    A Hook Card captures a small, traceable follow-up for new content needs,
    plot threads, or uncertainties discovered during creative work. Hooks are
    the entry point for new work in QuestFoundry's loop system.

    Workflow:
        1. Created by any role when encountering uncertainty or new need
        2. Classified by Showrunner (beat expansion, research, asset, etc.)
        3. Routed to appropriate loop (manuscript, art, audio, or research)
        4. Opened into TUBrief for execution
        5. Tracked throughout lifecycle until resolved

    Key characteristics:
        - Player-safe: Contains no spoilers or internal plumbing
        - Lightweight: Captures just enough context to route and prioritize
        - Traceable: Has unique ID and tracks resolution status
        - Classified: Tagged with hook type for appropriate loop routing

    Typical data fields:
        - header: {short_name, tags, priority}
        - summary: Brief description of the need
        - context: Where/why this hook arose
        - classification: Hook type (beat_expansion, research, etc.)
        - status: pending, opened, resolved, closed

    Example:
        >>> hook = HookCard(
        ...     data={
        ...         "header": {
        ...             "short_name": "Dragon Sighting Details",
        ...             "tags": ["worldbuilding", "dragon"],
        ...             "priority": "medium"
        ...         },
        ...         "summary": "Expand dragon appearance and reaction",
        ...         "classification": "beat_expansion"
        ...     },
        ...     metadata={"id": "HOOK-042"}
        ... )
    """

    type: str = "hook_card"


class TUBrief(Artifact):
    """
    Thematic Unit (TU) Brief artifact.

    A TU Brief defines a discrete unit of work to be executed within a loop.
    It represents the "opening" of a Hook into actionable work with clear scope,
    success criteria, and role assignments.

    The TU Brief serves as the contract between Showrunner (who opens TUs) and
    the roles that execute them (Writer, Artist, Archivist, etc.). It ensures
    work is well-scoped, traceable, and completes with clear deliverables.

    Lifecycle:
        1. Showrunner opens TU from a Hook
        2. Appropriate roles are woken with scoped tasks
        3. Roles execute work and produce artifacts
        4. Showrunner reviews completeness
        5. TU is closed when success criteria met
        6. Results merged to cold storage after gatecheck

    Key components:
        - Scope: What needs to be done (from originating Hook)
        - Success criteria: Definition of done
        - Role assignments: Which roles need to wake and what they should do
        - Deliverables: Expected artifact outputs
        - Context: Background and constraints

    Typical data fields:
        - header: {short_name, tu_id, opened_from_hook}
        - scope: Clear boundaries of what's in/out of scope
        - success_criteria: Checklist of completion requirements
        - wake_roles: List of roles and their specific tasks
        - deliverables: Expected artifact types to produce
        - context: Background information and constraints

    Example:
        >>> tu_brief = TUBrief(
        ...     data={
        ...         "header": {
        ...             "short_name": "Expand Tavern Scene",
        ...             "tu_id": "TU-123",
        ...             "opened_from_hook": "HOOK-042"
        ...         },
        ...         "scope": "Add dragon sighting dialogue and reactions",
        ...         "success_criteria": [
        ...             "Dialogue written for 3 NPCs",
        ...             "Player choices added",
        ...             "Codex entry drafted"
        ...         ],
        ...         "wake_roles": ["writer", "archivist"]
        ...     },
        ...     metadata={"id": "TU-123"}
        ... )
    """

    type: str = "tu_brief"


class CanonPack(Artifact):
    """
    Canon Pack artifact.

    A Canon Pack contains authoritative lore, worldbuilding facts, and
    narrative truth for the project. It serves as the "source of truth"
    for consistent worldbuilding across all content.

    Canon Packs are maintained by the Archivist role and referenced by
    other roles (Writer, Researcher) to ensure consistency. They bridge
    the gap between creative ideation and systematic world maintenance.

    Purpose:
        - Establish canonical facts (geography, history, character backgrounds)
        - Provide reference for consistent worldbuilding
        - Track what's been established vs. what's still flexible
        - Enable contradiction detection and consistency checks
        - Support player-facing codex generation

    Content types:
        - Worldbuilding facts (geography, history, cultures)
        - Character backgrounds and relationships
        - Magic/tech system rules and constraints
        - Timeline and chronology
        - Terminology and naming conventions
        - Internal notes and author guidance

    Typical data fields:
        - header: {short_name, category, canon_level}
        - facts: List of canonical statements
        - references: Links to source manuscripts/notes
        - relationships: Connections to other canon
        - visibility: Which facts are player-facing vs. internal
        - flexibility: What's locked vs. still negotiable

    Canon levels:
        - locked: Established in published/cold content, cannot change
        - soft: Established but can be refined if needed
        - provisional: Working assumption, subject to change
        - speculation: Possibility being explored

    Example:
        >>> canon = CanonPack(
        ...     data={
        ...         "header": {
        ...             "short_name": "Dragon Lore",
        ...             "category": "creatures",
        ...             "canon_level": "locked"
        ...         },
        ...         "facts": [
        ...             {
        ...                 "statement": "Dragons sleep for decades between hunts",
        ...                 "visibility": "player_facing",
        ...                 "established_in": "MS-ACT1-003"
        ...             }
        ...         ],
        ...         "relationships": ["REF-MAGIC-SYSTEM"]
        ...     },
        ...     metadata={"id": "CANON-DRAGONS-001"}
        ... )
    """

    type: str = "canon_pack"


class GatecheckReport(Artifact):
    """
    Gatecheck Report artifact.

    A Gatecheck Report documents the results of quality bar validation
    performed by the Gatekeeper role. It determines whether content is
    ready to promote from Hot (work-in-progress) to Cold (ship-ready).

    The Gatekeeper runs multiple quality bars (integrity, reachability,
    style, spoiler hygiene, etc.) and aggregates results into a single
    pass/fail decision with detailed issue tracking.

    Quality bars evaluated:
        - Integrity: References resolve, no unintended dead ends
        - Reachability: Keystone content reachable from start
        - Style: Voice, register, and motifs consistent
        - Gateways: Conditions diegetic and enforceable
        - Nonlinearity: Hubs/loops/choices meaningful
        - Determinism: Assets reproducible from parameters
        - Presentation: No spoilers/internals in player-facing text
        - Spoiler Hygiene: PN boundaries maintained

    Report structure:
        - Overall pass/fail status
        - Individual bar results with issues
        - Blockers (must fix before promotion)
        - Warnings (should review)
        - Info (optional improvements)
        - Promotion recommendation

    Typical data fields:
        - header: {timestamp, artifacts_checked, overall_status}
        - bar_results: List of QualityBarResult objects
        - blockers: Issues preventing promotion
        - warnings: Issues requiring review
        - summary: Human-readable assessment
        - recommendation: "promote", "revise", or "block"

    Workflow:
        1. Gatekeeper receives promotion request for artifacts
        2. Runs all applicable quality bars
        3. Aggregates issues by severity
        4. Generates report with recommendation
        5. If passed, artifacts promoted to cold
        6. If blocked, report returned to authors for revision

    Example:
        >>> report = GatecheckReport(
        ...     data={
        ...         "header": {
        ...             "timestamp": "2025-01-15T10:30:00Z",
        ...             "artifacts_checked": 5,
        ...             "overall_status": "passed"
        ...         },
        ...         "bar_results": [
        ...             {
        ...                 "bar_name": "integrity",
        ...                 "passed": True,
        ...                 "issues": []
        ...             },
        ...             {
        ...                 "bar_name": "style",
        ...                 "passed": True,
        ...                 "issues": [
        ...                     {
        ...                         "severity": "info",
        ...                         "message": "Consider consistent tense"
        ...                     }
        ...                 ]
        ...             }
        ...         ],
        ...         "recommendation": "promote"
        ...     },
        ...     metadata={"id": "GATE-2025-001"}
        ... )
    """

    type: str = "gatecheck_report"


class CodexEntry(Artifact):
    """
    Codex Entry artifact.

    Player-facing encyclopedia entry.
    """

    type: str = "codex_entry"


class StyleAddendum(Artifact):
    """
    Style Addendum artifact.

    Style guide additions or modifications.
    """

    type: str = "style_addendum"


class ResearchMemo(Artifact):
    """
    Research Memo artifact.

    Research findings and references.
    """

    type: str = "research_memo"


class Shotlist(Artifact):
    """
    Shotlist artifact.

    Visual composition and scene direction notes.
    """

    type: str = "shotlist"


class Cuelist(Artifact):
    """
    Cuelist artifact.

    Audio cue timing and direction notes.
    """

    type: str = "cuelist"


class ViewLog(Artifact):
    """
    View Log artifact.

    Record of view generation and export operations.
    """

    type: str = "view_log"


class ArtPlan(Artifact):
    """
    Art Plan artifact.

    Planning document for visual art direction.
    """

    type: str = "art_plan"


class ArtManifest(Artifact):
    """
    Art Manifest artifact.

    Inventory of art assets.
    """

    type: str = "art_manifest"


class AudioPlan(Artifact):
    """
    Audio Plan artifact.

    Planning document for audio direction.
    """

    type: str = "audio_plan"


class EditNotes(Artifact):
    """
    Edit Notes artifact.

    Editorial feedback and revision notes.
    """

    type: str = "edit_notes"


class FrontMatter(Artifact):
    """
    Front Matter artifact.

    Book front matter content (title page, copyright, etc.).
    """

    type: str = "front_matter"


class LanguagePack(Artifact):
    """
    Language Pack artifact.

    Translated content for a specific language.
    """

    type: str = "language_pack"


class PNPlaytestNotes(Artifact):
    """
    PN Playtest Notes artifact.

    Player Narrator feedback from playtesting sessions.
    """

    type: str = "pn_playtest_notes"


class ProjectMetadata(Artifact):
    """
    Project Metadata artifact.

    Top-level project configuration, metadata, and settings. This artifact
    defines global project parameters, versioning, authorship, and workflow
    configuration for the entire QuestFoundry project.

    ProjectMetadata serves as the "control panel" for a QuestFoundry project,
    containing essential information for initialization, collaboration, and
    export/distribution.

    Key configuration areas:
        - Project identity (name, version, description)
        - Authorship and attribution
        - Workflow settings (quality bar thresholds, loop preferences)
        - Export configuration (formats, filters, destinations)
        - Integration settings (provider credentials, external tools)
        - Style guidelines (default voice, register)
        - Versioning and changelog

    Typical data fields:
        - project_name: Human-readable project name
        - version: Semantic version (e.g., "0.1.0")
        - description: Project summary
        - authors: List of contributors
        - created: Project creation date
        - last_modified: Last update timestamp
        - workflow_config: Quality bar settings, loop preferences
        - export_config: Output formats and destinations
        - style_defaults: Default voice, register, themes
        - integrations: Provider API keys and settings

    Usage:
        ProjectMetadata is typically loaded at workspace initialization and
        used to configure the WorkspaceManager, roles, and export systems.

    Example:
        >>> metadata = ProjectMetadata(
        ...     data={
        ...         "project_name": "The Dragon's Gambit",
        ...         "version": "0.3.0",
        ...         "description": "Interactive fantasy adventure",
        ...         "authors": ["Jane Writer", "John Artist"],
        ...         "workflow_config": {
        ...             "quality_bars": ["integrity", "reachability", "style"],
        ...             "auto_promote": False
        ...         },
        ...         "export_config": {
        ...             "formats": ["html", "epub"],
        ...             "player_safe_only": True
        ...         },
        ...         "style_defaults": {
        ...             "voice": "epic fantasy",
        ...             "register": "literary"
        ...         }
        ...     },
        ...     metadata={"id": "PROJECT-META"}
        ... )
    """

    type: str = "project_metadata"


class RegisterMap(Artifact):
    """
    Register Map artifact.

    Language register and tone mapping for characters/scenes.
    """

    type: str = "register_map"


class StyleManifest(Artifact):
    """
    Style Manifest artifact.

    Master style guide inventory.
    """

    type: str = "style_manifest"


class CanonTransferPackage(Artifact):
    """
    Canon Transfer Package artifact.

    A Canon Transfer Package exports canon elements for reuse in sequels or
    shared universe projects. It packages worldbuilding, timeline anchors,
    entity registries, and codex baselines for import into new projects.

    Purpose:
        - Enable canon continuity across multiple projects/sequels
        - Share worldbuilding in collaborative/shared universes
        - Prevent contradictions when extending established worlds
        - Preserve immutable canon while allowing extensions

    Canon mutability levels:
        - **Invariant**: Immutable rules that cannot be changed (core canon)
        - **Mutable**: Extensible canon that can be elaborated upon
        - **Local**: Project-specific canon not exported

    Package contents:
        - Invariant canon packs (immutable worldbuilding rules)
        - Mutable canon packs (extensible worldbuilding)
        - Codex baseline (player-facing reference)
        - Timeline anchors (T0/T1/T2 baseline events)
        - Entity registry (characters, places, factions, items)
        - Constraint documentation (creative boundaries)

    Typical data fields:
        - header: {package_id, source_project, created_by, version}
        - invariant_canon: List of immutable canon pack references
        - mutable_canon: List of extensible canon pack references
        - codex_baseline: Player-facing reference entries
        - timeline: {anchors: [T0, T1, T2, ...], offsets: {...}}
        - entity_registry: {
            characters: [{name, role, description, source}],
            places: [{name, type, description, source}],
            factions: [{name, type, description, source}],
            items: [{name, type, description, source}]
          }
        - constraints: Manifest of invariants, mutables, timeline, entities
        - metadata: Export date, gatecheck status, compatibility version

    Workflow:
        1. Complete project reaches final Binding Run
        2. Author tags canon elements (invariant/mutable/local)
        3. System extracts tagged canon, codex, timeline, entities
        4. Gatekeeper validates for spoilers, conflicts, broken references
        5. Package exported as `canon_transfer_package_<slug>.json`
        6. Import into new project with conflict detection
        7. Merge to Cold (invariant) or Hot (mutable) storage

    Import conflict resolution:
        When invariant canon conflicts with new project seed:
        - **Reject**: Abandon conflicting seed idea, keep invariant
        - **Revise**: Modify seed to align with invariant canon
        - **Downgrade**: Change invariant to mutable, allow seed

    Example:
        >>> package = CanonTransferPackage(
        ...     data={
        ...         "header": {
        ...             "package_id": "TRANSFER-DRAGONS-001",
        ...             "source_project": "dragon-quest-1",
        ...             "created_by": "Author",
        ...             "version": "1.0.0"
        ...         },
        ...         "invariant_canon": [
        ...             "CANON-DRAGONS-001",  # Dragons sleep for decades
        ...             "CANON-MAGIC-SYSTEM-001"  # Magic requires sacrifice
        ...         ],
        ...         "mutable_canon": [
        ...             "CANON-GEOGRAPHY-001",  # Map can be extended
        ...             "CANON-FACTIONS-001"  # New factions can emerge
        ...         ],
        ...         "timeline": {
        ...             "anchors": [
        ...                 {"id": "T0", "description": "Founding of Kingdom"},
        ...                 {"id": "T1", "description": "Dragon Wars begin"},
        ...                 {"id": "T2", "description": "Peace treaty signed"}
        ...             ]
        ...         },
        ...         "entity_registry": {
        ...             "characters": [
        ...                 {
        ...                     "name": "Queen Elara",
        ...                     "role": "ruler",
        ...                     "description": "First queen of united kingdom",
        ...                     "source": "dragon-quest-1"
        ...                 }
        ...             ],
        ...             "places": [
        ...                 {
        ...                     "name": "Dragon's Spine Mountains",
        ...                     "type": "geographical",
        ...                     "description": "Impassable mountain range",
        ...                     "source": "dragon-quest-1"
        ...                 }
        ...             ]
        ...         }
        ...     },
        ...     metadata={"id": "TRANSFER-001"}
        ... )
    """

    type: str = "canon_transfer_package"


class WorldGenesisManifest(Artifact):
    """
    World Genesis Manifest artifact.

    A World Genesis Manifest documents proactive worldbuilding performed
    before story development begins. It captures the output of Canon-First
    workflow where worldbuilding themes are explored systematically to
    establish a rich, coherent universe before writing begins.

    Purpose:
        - Enable "Canon-First" project initialization workflow
        - Document systematic worldbuilding by theme
        - Establish constraints before Story Spark begins
        - Track worldbuilding iteration and stabilization
        - Provide foundation for multiple stories in same universe

    Worldbuilding budgets:
        - **Minimal** (2-4 hours): Core themes only, light detail
        - **Standard** (5-10 hours): Balanced coverage, moderate depth
        - **Epic** (20+ hours): Exhaustive themes, deep lore

    Common worldbuilding themes:
        - Geography & climate
        - History & major events
        - Cultures & societies
        - Magic/technology systems
        - Factions & power structures
        - Economy & resources
        - Religion & beliefs
        - Language & communication

    Typical data fields:
        - header: {manifest_id, project_slug, budget, created_date}
        - themes: List of worldbuilding themes explored
        - canon_packs: {theme_name: [canon_pack_ids]}
        - codex_baseline: Initial player-facing reference entries
        - style_addendum: Voice, register, tone for this universe
        - timeline: Chronological anchor points
        - constraints: {
            invariants: [immutable rules],
            mutables: [extensible rules],
            boundaries: [creative limits]
          }
        - entity_registry: Initial characters, places, factions, items
        - iteration_history: [{phase, artifacts_created, duration}]
        - stabilization_status: "unstable" | "converging" | "stable"

    Workflow:
        1. Author provides project concept and theme selection
        2. System runs Lore Deepening proactively per theme
        3. Researcher corroborates facts, checks contradictions
        4. Archivist builds canon packs and codex baseline
        5. Stylist establishes voice and register guidelines
        6. Gatekeeper validates for completeness and consistency
        7. Package merged to Cold storage
        8. Constraint manifest generated for creative roles
        9. Ready for Story Spark (plot development begins)

    Integration with Story Spark:
        - Constraint manifest displayed in sidebar
        - Plot changes validated against invariants
        - Entity list auto-populated from registry
        - Timeline anchors enforce chronology
        - Canon packs provide reference context

    Example:
        >>> manifest = WorldGenesisManifest(
        ...     data={
        ...         "header": {
        ...             "manifest_id": "GENESIS-SCIFI-001",
        ...             "project_slug": "star-colony",
        ...             "budget": "standard",
        ...             "created_date": "2025-01-15"
        ...         },
        ...         "themes": [
        ...             "solar-system-geography",
        ...             "ftl-technology",
        ...             "colonial-factions",
        ...             "alien-first-contact"
        ...         ],
        ...         "canon_packs": {
        ...             "solar-system-geography": [
        ...                 "CANON-SOL-SYSTEM-001",
        ...                 "CANON-MARS-COLONY-001"
        ...             ],
        ...             "ftl-technology": [
        ...                 "CANON-WARP-DRIVE-001"
        ...             ]
        ...         },
        ...         "constraints": {
        ...             "invariants": [
        ...                 "FTL travel requires antimatter fuel",
        ...                 "Mars colony established 2147 AD"
        ...             ],
        ...             "mutables": [
        ...                 "New colonies can be founded",
        ...                 "Alien species traits not fully defined"
        ...             ]
        ...         },
        ...         "timeline": {
        ...             "anchors": [
        ...                 {"id": "T0", "year": 2147, "event": "Mars Colony Founded"},
        ...                 {"id": "T1", "year": 2203, "event": "FTL Discovered"}
        ...             ]
        ...         },
        ...         "stabilization_status": "stable"
        ...     },
        ...     metadata={"id": "GENESIS-001"}
        ... )
    """

    type: str = "world_genesis_manifest"
