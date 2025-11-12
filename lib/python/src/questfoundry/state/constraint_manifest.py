"""
Constraint manifest generation for canon workflows.

Generates constraint documentation for creative roles, outlining invariants,
mutables, timeline constraints, and entity registries. Used to guide Story Spark
and other creative loops while maintaining canon integrity.
"""

from dataclasses import dataclass, field
from typing import Any

from .entity_registry import EntityRegistry, EntityType
from .timeline import TimelineManager


@dataclass
class ConstraintManifest:
    """
    Manifest of creative constraints derived from canon.

    The constraint manifest provides clear boundaries for creative roles
    (Plotwright, Scene Smith, etc.) by documenting what can and cannot be
    changed based on invariant and mutable canon.

    Attributes:
        invariants: List of immutable canon rules ("You CANNOT")
        mutables: List of extensible canon elements ("You CAN")
        timeline_constraints: Chronological boundaries
        entity_constraints: Restrictions on canonical entities
        boundaries: Overall creative boundaries
        guidance: Positive guidance for Story Spark
        metadata: Additional manifest data
    """

    invariants: list[str] = field(default_factory=list)
    mutables: list[str] = field(default_factory=list)
    timeline_constraints: list[str] = field(default_factory=list)
    entity_constraints: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "invariants": self.invariants,
            "mutables": self.mutables,
            "timeline_constraints": self.timeline_constraints,
            "entity_constraints": self.entity_constraints,
            "boundaries": self.boundaries,
            "guidance": self.guidance,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """
        Generate human-readable markdown documentation.

        Returns:
            Markdown-formatted constraint documentation
        """
        sections = []

        sections.append("# Creative Constraints")
        sections.append("")
        sections.append(
            "This document outlines the creative boundaries for this project, "
            "derived from canon transfer packages and world genesis."
        )
        sections.append("")

        # Invariants section
        if self.invariants:
            sections.append("## You CANNOT (Invariant Canon)")
            sections.append("")
            sections.append(
                "These rules are immutable and cannot be changed or contradicted:"
            )
            sections.append("")
            for i, invariant in enumerate(self.invariants, 1):
                sections.append(f"{i}. ❌ {invariant}")
            sections.append("")

        # Mutables section
        if self.mutables:
            sections.append("## You CAN (Mutable Canon)")
            sections.append("")
            sections.append("These elements can be extended, elaborated, or refined:")
            sections.append("")
            for i, mutable in enumerate(self.mutables, 1):
                sections.append(f"{i}. ✅ {mutable}")
            sections.append("")

        # Timeline section
        if self.timeline_constraints:
            sections.append("## Timeline Constraints")
            sections.append("")
            sections.append("Chronological boundaries that must be respected:")
            sections.append("")
            for constraint in self.timeline_constraints:
                sections.append(f"- 📅 {constraint}")
            sections.append("")

        # Entity section
        if self.entity_constraints:
            sections.append("## Entity Constraints")
            sections.append("")
            sections.append("Restrictions on canonical entities:")
            sections.append("")
            for constraint in self.entity_constraints:
                sections.append(f"- 👤 {constraint}")
            sections.append("")

        # Boundaries section
        if self.boundaries:
            sections.append("## Creative Boundaries")
            sections.append("")
            for boundary in self.boundaries:
                sections.append(f"- ⚠️ {boundary}")
            sections.append("")

        # Guidance section
        if self.guidance:
            sections.append("## Creative Guidance")
            sections.append("")
            sections.append("Positive guidance for story development:")
            sections.append("")
            for guide in self.guidance:
                sections.append(f"- 💡 {guide}")
            sections.append("")

        return "\n".join(sections)


class ConstraintManifestGenerator:
    """
    Generator for constraint manifests.

    Creates constraint documentation from canon packs, entity registries,
    and timeline anchors. Used during canon import and world genesis to
    provide clear creative boundaries.

    Example:
        >>> from questfoundry.models import CanonPack
        >>> generator = ConstraintManifestGenerator()
        >>> # Add invariant canon
        >>> invariant_canon = [
        ...     CanonPack(data={
        ...         "facts": ["Dragons sleep for decades between hunts"],
        ...         "immutable": True
        ...     })
        ... ]
        >>> # Add entity registry
        >>> registry = EntityRegistry()
        >>> registry.create(Entity(
        ...     name="Queen Elara",
        ...     entity_type=EntityType.CHARACTER,
        ...     role="ruler",
        ...     description="First queen of united kingdom",
        ...     source="world-genesis",
        ...     immutable=True
        ... ))
        >>> # Generate manifest
        >>> manifest = generator.generate(
        ...     invariant_canon=invariant_canon,
        ...     entity_registry=registry
        ... )
        >>> print(manifest.to_markdown())
    """

    def generate(
        self,
        invariant_canon: list[dict[str, Any]] | None = None,
        mutable_canon: list[dict[str, Any]] | None = None,
        entity_registry: EntityRegistry | None = None,
        timeline: TimelineManager | None = None,
        source: str = "canon-import",
    ) -> ConstraintManifest:
        """
        Generate constraint manifest from canon components.

        Args:
            invariant_canon: List of immutable canon pack data
            mutable_canon: List of extensible canon pack data
            entity_registry: Entity registry with canonical entities
            timeline: Timeline manager with chronological anchors
            source: Source of the canon (for metadata)

        Returns:
            ConstraintManifest with all constraints
        """
        manifest = ConstraintManifest()
        manifest.metadata["source"] = source

        # Process invariant canon
        if invariant_canon:
            for canon in invariant_canon:
                manifest.invariants.extend(self._extract_invariants(canon))

        # Process mutable canon
        if mutable_canon:
            for canon in mutable_canon:
                manifest.mutables.extend(self._extract_mutables(canon))

        # Process entity constraints
        if entity_registry:
            manifest.entity_constraints.extend(
                self._extract_entity_constraints(entity_registry)
            )

        # Process timeline constraints
        if timeline:
            manifest.timeline_constraints.extend(
                self._extract_timeline_constraints(timeline)
            )

        # Generate boundaries
        manifest.boundaries = self._generate_boundaries(
            len(manifest.invariants),
            len(manifest.entity_constraints),
            len(manifest.timeline_constraints),
        )

        # Generate positive guidance
        manifest.guidance = self._generate_guidance(
            len(manifest.mutables),
            entity_registry,
        )

        return manifest

    def _extract_invariants(self, canon: dict[str, Any]) -> list[str]:
        """Extract invariant rules from canon pack."""
        invariants = []

        # Extract facts marked as immutable
        if "facts" in canon:
            facts = canon["facts"]
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, str):
                        invariants.append(fact)
                    elif isinstance(fact, dict) and fact.get("immutable", True):
                        statement = fact.get("statement", "")
                        if statement:
                            invariants.append(statement)

        return invariants

    def _extract_mutables(self, canon: dict[str, Any]) -> list[str]:
        """Extract mutable elements from canon pack."""
        mutables = []

        # Extract facts marked as mutable
        if "facts" in canon:
            facts = canon["facts"]
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, dict) and not fact.get("immutable", True):
                        statement = fact.get("statement", "")
                        if statement:
                            # Rephrase as permission
                            mutables.append(f"Extend or elaborate on: {statement}")

        return mutables

    def _extract_entity_constraints(self, registry: EntityRegistry) -> list[str]:
        """Extract entity constraints from registry."""
        constraints = []

        # Immutable characters
        immutable_chars = [
            e for e in registry.get_by_type(EntityType.CHARACTER) if e.immutable
        ]
        if immutable_chars:
            names = [e.name for e in immutable_chars[:5]]  # First 5
            count_more = len(immutable_chars) - 5
            constraint = f"Cannot modify canonical characters: {', '.join(names)}"
            if count_more > 0:
                constraint += f" (and {count_more} more)"
            constraints.append(constraint)

        # Immutable places
        immutable_places = [
            e for e in registry.get_by_type(EntityType.PLACE) if e.immutable
        ]
        if immutable_places:
            names = [e.name for e in immutable_places[:5]]
            count_more = len(immutable_places) - 5
            constraint = f"Cannot modify canonical locations: {', '.join(names)}"
            if count_more > 0:
                constraint += f" (and {count_more} more)"
            constraints.append(constraint)

        return constraints

    def _extract_timeline_constraints(self, timeline: TimelineManager) -> list[str]:
        """Extract timeline constraints."""
        constraints = []

        # Baseline anchors
        baseline = timeline.get_baseline_anchors()
        for anchor in baseline:
            if anchor.immutable:
                constraint = f"{anchor.anchor_id}: {anchor.event}"
                if anchor.year:
                    constraint += f" ({anchor.year})"
                constraints.append(constraint)

        # Add general chronology constraint
        if baseline:
            t0 = timeline.get_anchor("T0")
            if t0 and t0.year:
                constraints.append(
                    f"All events must occur after {t0.year} (T0 baseline)"
                )

        return constraints

    def _generate_boundaries(
        self,
        invariant_count: int,
        entity_count: int,
        timeline_count: int,
    ) -> list[str]:
        """Generate overall creative boundaries."""
        boundaries = []

        if invariant_count > 0:
            boundaries.append(f"Respect {invariant_count} immutable canon rule(s)")

        if entity_count > 0:
            boundaries.append(
                f"Maintain consistency with {entity_count} "
                "canonical entity constraint(s)"
            )

        if timeline_count > 0:
            boundaries.append(f"Adhere to {timeline_count} timeline constraint(s)")

        if not boundaries:
            boundaries.append("No strict constraints - full creative freedom")

        return boundaries

    def _generate_guidance(
        self,
        mutable_count: int,
        entity_registry: EntityRegistry | None,
    ) -> list[str]:
        """Generate positive creative guidance."""
        guidance = []

        if mutable_count > 0:
            guidance.append(
                f"Freely extend and elaborate on {mutable_count} "
                "mutable canon element(s)"
            )

        if entity_registry:
            counts = entity_registry.count_by_type()
            guidance.append(
                f"Reference established entities: {counts['characters']} characters, "
                f"{counts['places']} places, {counts['factions']} factions, "
                f"{counts['items']} items"
            )

        guidance.append("Introduce new entities, events, and details as needed")
        guidance.append("Maintain consistency with established tone and style")

        return guidance
