"""Loop registry for QuestFoundry."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoopMetadata:
    """
    Lightweight loop description for selection.

    This provides just enough information for the Showrunner to make
    strategic decisions about which loop to run without loading full
    loop implementation details.
    """

    loop_id: str
    """Unique loop identifier (e.g., 'story_spark')"""

    display_name: str
    """Human-readable name (e.g., 'Story Spark')"""

    description: str
    """One-line purpose statement"""

    typical_duration: str
    """Expected duration (e.g., '2-4 hours')"""

    primary_roles: list[str] = field(default_factory=list)
    """Main roles involved (RACI: Responsible)"""

    consulted_roles: list[str] = field(default_factory=list)
    """Supporting roles (RACI: Consulted)"""

    entry_conditions: list[str] = field(default_factory=list)
    """When this loop should be triggered"""

    exit_conditions: list[str] = field(default_factory=list)
    """What marks successful completion"""

    output_artifacts: list[str] = field(default_factory=list)
    """Expected artifact types produced"""

    inputs: list[str] = field(default_factory=list)
    """Required inputs from prior work"""

    tags: list[str] = field(default_factory=list)
    """Categorization tags (e.g., 'structure', 'quality', 'content')"""


class LoopRegistry:
    """
    Registry of all available loops.

    Provides lightweight metadata for loop selection without loading
    full implementation details. This enables the Showrunner to make
    strategic decisions with minimal context (~90 lines total).
    """

    def __init__(self, spec_path: Path | None = None):
        """
        Initialize loop registry.

        Args:
            spec_path: Path to spec directory (default: ./spec)
        """
        self.spec_path = spec_path or Path.cwd() / "spec"
        self._loops: dict[str, LoopMetadata] = {}
        self._register_builtin_loops()

    def _register_builtin_loops(self) -> None:
        """Register all built-in loop metadata."""

        # 1. Story Spark - Initial quest concept to first draft
        self.register_loop(
            LoopMetadata(
                loop_id="story_spark",
                display_name="Story Spark",
                description="Introduce or reshape narrative structure",
                typical_duration="2-4 hours",
                primary_roles=["plotwright", "scene_smith"],
                consulted_roles=[
                    "style_lead",
                    "lore_weaver",
                    "codex_curator",
                    "gatekeeper",
                ],
                entry_conditions=[
                    "New chapter/act/subplot needed",
                    "Restructure request",
                    "Reachability/nonlinearity fixes",
                ],
                exit_conditions=[
                    "Topology stabilized",
                    "Section briefs complete",
                    "Scene drafts ready",
                    "Gatekeeper preview passed",
                ],
                output_artifacts=["tu_brief", "hook_card", "canon_pack"],
                inputs=[
                    "Cold snapshot",
                    "Prior topology notes",
                    "Open hooks",
                    "QA findings",
                ],
                tags=["structure", "content", "foundation"],
            )
        )

        # 2. Hook Harvest - Generate and refine quest hooks
        self.register_loop(
            LoopMetadata(
                loop_id="hook_harvest",
                display_name="Hook Harvest",
                description="Collect, cluster, and triage hooks",
                typical_duration="1-2 hours",
                primary_roles=["showrunner"],
                consulted_roles=[
                    "lore_weaver",
                    "plotwright",
                    "scene_smith",
                    "codex_curator",
                    "style_lead",
                    "gatekeeper",
                ],
                entry_conditions=[
                    "After Story Spark or drafting burst",
                    "Before stabilization window",
                    "Backlog looks fuzzy",
                ],
                exit_conditions=[
                    "Hooks prioritized and tagged",
                    "Duplicates removed",
                    "Dependencies identified",
                    "Ready for Lore Deepening",
                ],
                output_artifacts=["hook_card"],
                inputs=[
                    "Hook cards in Hot",
                    "Topology notes",
                    "Section drafts",
                    "QA notes",
                ],
                tags=["planning", "organization"],
            )
        )

        # 3. Lore Deepening - Expand world-building
        self.register_loop(
            LoopMetadata(
                loop_id="lore_deepening",
                display_name="Lore Deepening",
                description="Canonize hooks into world lore",
                typical_duration="2-3 hours",
                primary_roles=["lore_weaver"],
                consulted_roles=[
                    "researcher",
                    "codex_curator",
                    "plotwright",
                    "gatekeeper",
                ],
                entry_conditions=[
                    "After Hook Harvest",
                    "Canon gaps identified",
                    "World-building expansion needed",
                ],
                exit_conditions=[
                    "Hooks canonized",
                    "Lore consistency maintained",
                    "Codex hooks generated",
                ],
                output_artifacts=["canon_pack", "hook_card"],
                inputs=["Prioritized hooks", "Cold canon", "Codex entries"],
                tags=["world-building", "lore"],
            )
        )

        # 4. Codex Expansion - Document lore entries
        self.register_loop(
            LoopMetadata(
                loop_id="codex_expansion",
                display_name="Codex Expansion",
                description="Create player-safe codex entries",
                typical_duration="1-2 hours",
                primary_roles=["codex_curator"],
                consulted_roles=["lore_weaver", "style_lead", "gatekeeper"],
                entry_conditions=[
                    "After Lore Deepening",
                    "Taxonomy gaps identified",
                    "Player-facing documentation needed",
                ],
                exit_conditions=[
                    "Codex entries complete",
                    "Player-neutral verified",
                    "Taxonomy updated",
                ],
                output_artifacts=["codex_entry"],
                inputs=["Canon packs", "Existing codex", "Taxonomy map"],
                tags=["documentation", "player-facing"],
            )
        )

        # 5. Binding Run - Assemble complete quest package
        self.register_loop(
            LoopMetadata(
                loop_id="binding_run",
                display_name="Binding Run",
                description="Assemble and finalize quest package",
                typical_duration="2-3 hours",
                primary_roles=["book_binder"],
                consulted_roles=["gatekeeper", "style_lead"],
                entry_conditions=[
                    "All content complete",
                    "Gate checks passed",
                    "Ready for export",
                ],
                exit_conditions=[
                    "Package assembled",
                    "Front matter complete",
                    "Export ready",
                ],
                output_artifacts=["front_matter", "view_log"],
                inputs=["All artifacts", "Gate check reports", "Style manifest"],
                tags=["finalization", "export"],
            )
        )

        # 6. Style Tune-Up - Refine writing style consistency
        self.register_loop(
            LoopMetadata(
                loop_id="style_tune_up",
                display_name="Style Tune-Up",
                description="Polish style consistency and voice",
                typical_duration="1-2 hours",
                primary_roles=["style_lead", "scene_smith"],
                consulted_roles=["gatekeeper"],
                entry_conditions=[
                    "Style drift detected",
                    "Tone inconsistencies",
                    "Polish pass needed",
                ],
                exit_conditions=[
                    "Style consistency restored",
                    "Voice uniform",
                    "Presentation improved",
                ],
                output_artifacts=["style_addendum", "edit_notes"],
                inputs=["Section drafts", "Style manifest", "Previous addenda"],
                tags=["quality", "polish"],
            )
        )

        # 7. Art Touch-Up - Generate/refine visual assets
        self.register_loop(
            LoopMetadata(
                loop_id="art_touch_up",
                display_name="Art Touch-Up",
                description="Create and refine visual assets",
                typical_duration="2-4 hours",
                primary_roles=["art_director", "illustrator"],
                consulted_roles=["style_lead", "gatekeeper"],
                entry_conditions=[
                    "Visual assets needed",
                    "Art plan complete",
                    "Shotlist ready",
                ],
                exit_conditions=[
                    "Images generated",
                    "Art manifest updated",
                    "Visual consistency verified",
                ],
                output_artifacts=["art_manifest", "shotlist"],
                inputs=["Art plan", "Scene descriptions", "Style guidelines"],
                tags=["visual", "assets"],
            )
        )

        # 8. Audio Pass - Create audio content
        self.register_loop(
            LoopMetadata(
                loop_id="audio_pass",
                display_name="Audio Pass",
                description="Generate audio narration and effects",
                typical_duration="2-3 hours",
                primary_roles=["audio_producer"],
                consulted_roles=["player_narrator", "style_lead", "gatekeeper"],
                entry_conditions=[
                    "Audio content needed",
                    "Narration scripts ready",
                    "Audio plan complete",
                ],
                exit_conditions=[
                    "Audio assets generated",
                    "Cuelist complete",
                    "Quality verified",
                ],
                output_artifacts=["cuelist", "audio_plan"],
                inputs=["Scene text", "Audio plan", "Voice guidelines"],
                tags=["audio", "assets"],
            )
        )

        # 9. Translation Pass - Localization workflow
        self.register_loop(
            LoopMetadata(
                loop_id="translation_pass",
                display_name="Translation Pass",
                description="Localize content to target languages",
                typical_duration="3-5 hours",
                primary_roles=["translator"],
                consulted_roles=["style_lead", "gatekeeper"],
                entry_conditions=[
                    "Localization requested",
                    "Source content stable",
                    "Target language specified",
                ],
                exit_conditions=[
                    "Translation complete",
                    "Register map updated",
                    "Quality verified",
                ],
                output_artifacts=["language_pack", "register_map"],
                inputs=["Source content", "Style manifest", "Prior translations"],
                tags=["localization", "translation"],
            )
        )

        # 10. Narration Dry Run - Playtest preparation
        self.register_loop(
            LoopMetadata(
                loop_id="narration_dry_run",
                display_name="Narration Dry Run",
                description="Prepare and test player-facing narration",
                typical_duration="1-2 hours",
                primary_roles=["player_narrator"],
                consulted_roles=["scene_smith", "gatekeeper"],
                entry_conditions=[
                    "Content ready for playtest",
                    "Player-neutral verified",
                    "Narration needed",
                ],
                exit_conditions=[
                    "Playtest notes complete",
                    "Issues identified",
                    "Narration tested",
                ],
                output_artifacts=["pn_playtest_notes"],
                inputs=["Cold snapshot", "Codex entries", "Scene text"],
                tags=["testing", "player-facing"],
            )
        )

        # 11. Full Production Run - Complete end-to-end workflow
        self.register_loop(
            LoopMetadata(
                loop_id="full_production_run",
                display_name="Full Production Run",
                description="Complete end-to-end quest production",
                typical_duration="1-2 weeks",
                primary_roles=["showrunner"],
                consulted_roles=[
                    "plotwright",
                    "scene_smith",
                    "lore_weaver",
                    "codex_curator",
                    "gatekeeper",
                ],
                entry_conditions=[
                    "New quest from scratch",
                    "Complete production cycle",
                    "All roles available",
                ],
                exit_conditions=[
                    "Quest fully complete",
                    "All quality bars passed",
                    "Ready for release",
                ],
                output_artifacts=[
                    "tu_brief",
                    "hook_card",
                    "canon_pack",
                    "codex_entry",
                    "front_matter",
                ],
                inputs=["Project requirements", "Style guidelines", "World canon"],
                tags=["complete", "production"],
            )
        )

    def register_loop(self, metadata: LoopMetadata) -> None:
        """
        Register a loop in the registry.

        Args:
            metadata: Loop metadata to register
        """
        self._loops[metadata.loop_id] = metadata

    def get_loop_metadata(self, loop_id: str) -> LoopMetadata:
        """
        Get loop metadata by ID.

        Args:
            loop_id: Loop identifier

        Returns:
            Loop metadata

        Raises:
            KeyError: If loop not found
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop '{loop_id}' not registered")
        return self._loops[loop_id]

    def list_loops(self, filters: dict[str, Any] | None = None) -> list[LoopMetadata]:
        """
        List loops matching optional filters.

        Args:
            filters: Optional filter criteria:
                - tag: Filter by tag
                - role: Filter by involved role
                - duration: Filter by duration category

        Returns:
            List of matching loop metadata
        """
        loops = list(self._loops.values())

        if not filters:
            return loops

        # Filter by tag
        if "tag" in filters:
            tag = filters["tag"]
            loops = [loop for loop in loops if tag in loop.tags]

        # Filter by role
        if "role" in filters:
            role = filters["role"]
            loops = [
                loop
                for loop in loops
                if role in loop.primary_roles or role in loop.consulted_roles
            ]

        # Filter by duration (simplified - just check if substring matches)
        if "duration" in filters:
            duration = filters["duration"]
            loops = [loop for loop in loops if duration in loop.typical_duration]

        return loops

    def get_loops_by_role(self, role: str) -> list[LoopMetadata]:
        """
        Get all loops involving a specific role.

        Args:
            role: Role identifier

        Returns:
            List of loops where role is primary or consulted
        """
        return self.list_loops(filters={"role": role})

    def get_loops_by_tag(self, tag: str) -> list[LoopMetadata]:
        """
        Get all loops with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching loops
        """
        return self.list_loops(filters={"tag": tag})

    def build_registry_context(self) -> str:
        """
        Build lightweight context for loop selection.

        This creates a ~90 line summary of all loops suitable for
        Showrunner decision-making without overwhelming context.

        Returns:
            Formatted string describing all loops
        """
        lines = ["# Available Loops\n"]

        for loop in sorted(self._loops.values(), key=lambda x: x.loop_id):
            lines.append(f"## {loop.display_name} ({loop.loop_id})")
            lines.append(f"{loop.description}")
            lines.append(f"Duration: {loop.typical_duration}")
            lines.append(f"Primary: {', '.join(loop.primary_roles)}")
            lines.append(f"Triggers: {', '.join(loop.entry_conditions[:2])}...")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"LoopRegistry(loops={len(self._loops)})"
