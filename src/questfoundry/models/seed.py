"""Pydantic models for SEED stage output.

SEED is the triage stage that transforms expansive brainstorm material into
committed story structure. It curates entities, decides which answers
to explore as paths, creates consequences, and defines initial beats.

CRITICAL: PATH FREEZE - No new paths can be created after SEED.

See docs/design/00-spec.md for details.

Terminology (v5):
- dilemma: Binary dramatic questions
- path: Routes exploring specific answers to dilemmas
- answer: Possible resolutions to dilemmas
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Type aliases for clarity
EntityDisposition = Literal["retained", "cut"]
# PathTier has no direct Document 1/3 counterpart but serves a real purpose:
# dilemma scoring uses it to weight non-canonical paths, and context formatting
# displays it to guide LLM prose generation. Kept as-is.
PathTier = Literal["major", "minor"]
DilemmaEffect = Literal["advances", "reveals", "commits", "complicates"]
DilemmaRole = Literal["hard", "soft"]
EndingSalience = Literal["high", "low", "none"]
ResidueWeight = Literal["heavy", "light", "cosmetic"]
DilemmaOrdering = Literal["wraps", "concurrent", "serial"]
TemporalPosition = Literal["before_commit", "after_commit", "before_introduce", "after_introduce"]


class EntityDecision(BaseModel):
    """Entity curation decision from SEED.

    SEED receives entities from BRAINSTORM and decides which to retain
    for the story. Entities are the building blocks; not all brainstorm
    ideas make it into the final story.

    When retaining an entity that lacks a name from BRAINSTORM, SEED
    generates an appropriate canonical name based on the entity's
    concept and role in the story.

    Attributes:
        entity_id: Entity ID from BRAINSTORM.
        disposition: Whether to keep (retained) or discard (cut).
        name: Canonical display name for retained entities (required if missing from BRAINSTORM).
    """

    entity_id: str = Field(
        min_length=1, description="Entity ID from BRAINSTORM (references entity_id)"
    )
    disposition: EntityDisposition = Field(
        default="retained",
        description="Whether to keep or discard the entity",
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Canonical display name for this entity. Required for retained entities "
            "that don't already have a name from BRAINSTORM (e.g., 'Lady Beatrice Ashford', "
            "'The Gilded Compass Inn'). Leave absent if entity already has a name."
        ),
    )


class DilemmaDecision(BaseModel):
    """Dilemma exploration decision from SEED.

    Each dilemma has two answers. SEED decides which answers to
    explore as paths. The canonical answer is always explored (spine).
    Non-canonical answers become branches only if explicitly explored.

    The `explored` field records the LLM's *intent* - which answers it
    wanted to explore. Actual path existence is derived at runtime from the
    graph, not from this field. This separation allows pruning to drop paths
    without modifying the dilemma's stored intent.

    Development states (computed, not stored):
    - committed: Answer has a path in the graph
    - deferred: Answer in `explored` but no path (pruned)
    - latent: Answer not in `explored` (never intended for exploration)

    Attributes:
        dilemma_id: Dilemma ID from BRAINSTORM.
        explored: Answer IDs the LLM intended to explore as paths.
        unexplored: Answer IDs not explored (context for FILL shadows).
    """

    dilemma_id: str = Field(min_length=1, description="Dilemma ID from BRAINSTORM")
    explored: list[str] = Field(
        min_length=1,
        description="Answer IDs the LLM intended to explore as paths",
    )
    unexplored: list[str] = Field(
        default_factory=list,
        description="Answer IDs not explored (become shadows)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_considered_field(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old 'considered' field to 'explored'."""
        if isinstance(data, dict) and "considered" in data and "explored" not in data:
            data = dict(data)  # Avoid mutating input
            data["explored"] = data.pop("considered")
        if isinstance(data, dict) and "implicit" in data and "unexplored" not in data:
            data = dict(data)
            data["unexplored"] = data.pop("implicit")
        return data


class Consequence(BaseModel):
    """Narrative consequence of a path choice.

    Consequences bridge the gap between "what this path represents" (answer)
    and "how we track it" (codeword). GROW creates codewords to track when
    consequences become active.

    Attributes:
        consequence_id: Unique identifier for the consequence.
        path_id: Path this consequence belongs to.
        description: What happens narratively.
        narrative_effects: Story effects this implies.
    """

    consequence_id: str = Field(min_length=1, description="Unique identifier for this consequence")
    path_id: str = Field(min_length=1, description="Path this belongs to (references path_id)")
    description: str = Field(min_length=1, description="Narrative meaning of this path")
    narrative_effects: list[str] = Field(
        default_factory=list,
        description="Story effects this consequence implies (cascading impacts)",
    )


class Path(BaseModel):
    """Plot path exploring one answer from a dilemma.

    Paths are the core structural units of the branching story. Paths
    from the same dilemma are automatically exclusive (choosing one means
    not choosing the other).

    Path IDs use hierarchical format: path::dilemma_id__answer_id
    This embeds the parent dilemma in the ID, making the relationship explicit.

    Attributes:
        path_id: Unique identifier (format: path::dilemma_id__answer_id).
        name: Human-readable name.
        dilemma_id: The dilemma this path explores (derivable from path_id).
        answer_id: The specific answer this path explores.
        unexplored_answer_ids: IDs of unexplored answers (context for FILL).
        path_importance: Major paths interweave; minor paths support.
        description: What this path is about.
        consequence_ids: IDs of consequences for this path.
    """

    path_id: str = Field(
        min_length=1, description="Unique identifier (format: path::dilemma_id__answer_id)"
    )
    name: str = Field(min_length=1, description="Human-readable name")
    dilemma_id: str = Field(
        min_length=1, description="Dilemma this explores (references dilemma_id)"
    )
    answer_id: str = Field(min_length=1, description="Answer this explores (references answer_id)")
    unexplored_answer_ids: list[str] = Field(
        default_factory=list,
        description="IDs of unexplored answers from same dilemma (context for FILL)",
    )
    path_importance: PathTier = Field(
        description="Path importance: major (interweaves) or minor (supports)"
    )
    description: str = Field(min_length=1, description="What this path is about")
    consequence_ids: list[str] = Field(
        default_factory=list,
        description="Consequence IDs for this path (references consequence_id)",
    )
    pov_character: str | None = Field(
        default=None,
        description="Entity ID of the POV character for this path (overrides global protagonist)",
    )


class DilemmaImpact(BaseModel):
    """How a beat affects a dilemma.

    Each beat can impact one or more dilemmas, moving the story forward
    in various ways.

    Attributes:
        dilemma_id: Dilemma being impacted.
        effect: How the beat affects the dilemma.
        note: Explanation of the impact.
    """

    dilemma_id: str = Field(min_length=1, description="Dilemma being impacted")
    effect: DilemmaEffect = Field(description="How the beat affects the dilemma")
    note: str = Field(min_length=1, description="Explanation of the impact")


class TemporalHint(BaseModel):
    """Advisory placement hint for a beat relative to another dilemma.

    SEED creates beats for individual paths. A beat's position relative to
    its own dilemma is clear from its effect (advances/commits/etc.), but
    its position relative to *other* dilemmas is not yet determined. Temporal
    hints tell GROW where this beat should fall relative to other dilemmas'
    key moments.

    Hints are advisory — GROW resolves conflicts in favor of dilemma ordering
    relationships (wraps/serial/concurrent).

    See docs/design/document-3-ontology.md § Temporal Hints.

    Attributes:
        relative_to: The dilemma ID this hint is relative to.
        position: Where this beat should fall relative to the dilemma's key moment.
    """

    relative_to: str = Field(
        min_length=1,
        description="Dilemma ID this hint is relative to",
    )
    position: TemporalPosition = Field(
        description=(
            "Placement relative to the dilemma: before_commit, after_commit, "
            "before_introduce, after_introduce"
        ),
    )


class InitialBeat(BaseModel):
    """Initial beat created by SEED.

    Beats are narrative units belonging to one or more paths. SEED creates
    the initial beats for each path; GROW mutates and adds more.

    Attributes:
        id: Unique identifier for the beat.
        summary: What happens in this beat.
        paths: Path IDs this beat serves.
        dilemma_impacts: How this beat affects dilemmas.
        entities: Entity IDs present in this beat.
        location: Primary location entity ID.
        location_alternatives: Other valid locations (enables intersection flexibility).
        temporal_hint: Advisory placement relative to another dilemma (consumed by GROW).
    """

    beat_id: str = Field(min_length=1, description="Unique identifier for this beat")
    summary: str = Field(min_length=1, description="What happens in this beat")
    paths: list[str] = Field(
        min_length=1,
        description="Path IDs this beat serves",
    )
    dilemma_impacts: list[DilemmaImpact] = Field(
        default_factory=list,
        description="How this beat affects dilemmas",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Entity IDs present in this beat",
    )
    location: str | None = Field(
        default=None,
        description="Primary location entity ID",
    )
    location_alternatives: list[str] = Field(
        default_factory=list,
        description="Alternate location entities for intersection flexibility. "
        "Stored as flexibility edges in graph (Doc 3), not as node property.",
    )
    temporal_hint: TemporalHint | None = Field(
        default=None,
        description=(
            "Advisory placement relative to another dilemma's key moment. "
            "Consumed by GROW during interleaving; not carried forward."
        ),
    )


class DilemmaAnalysis(BaseModel):
    """Per-dilemma convergence classification and guidance from SEED.

    Serialized as Section 7 (after prune). Tells GROW how strictly
    to enforce path separation for this dilemma and where paths converge.

    Attributes:
        dilemma_id: References a dilemma from Section 2.
        dilemma_role: How strictly paths must stay separate (topology).
        ending_salience: How much this dilemma drives ending differentiation (prose layer).
        residue_weight: How much mid-story prose varies based on this dilemma (prose layer).
        payoff_budget: Minimum exclusive beats before convergence (>=2).
        reasoning: Chain-of-thought for the classification.
        convergence_point: Where this dilemma's paths physically converge.
        residue_note: Differences that persist after convergence.
    """

    dilemma_id: str = Field(min_length=1, description="Dilemma ID from Section 2")
    dilemma_role: DilemmaRole = Field(
        description="How strictly paths must stay separate (hard|soft)",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate deprecated field names and values.

        - ``convergence_policy`` → ``dilemma_role`` (field rename)
        - ``dilemma_role='flavor'`` → ``'soft'`` with ``residue_weight='cosmetic'``
        """
        if not isinstance(data, dict):
            return data
        # Field-name migration: convergence_policy → dilemma_role
        if "convergence_policy" in data and "dilemma_role" not in data:
            data["dilemma_role"] = data.pop("convergence_policy")
        # Value migration: flavor → soft
        if data.get("dilemma_role") == "flavor":
            import warnings

            warnings.warn(
                "dilemma_role='flavor' is deprecated — use 'soft' with "
                "residue_weight='cosmetic' instead (see ADR-013/019)",
                DeprecationWarning,
                stacklevel=2,
            )
            data["dilemma_role"] = "soft"
            if data.get("residue_weight") is None:
                data["residue_weight"] = "cosmetic"
        return data

    ending_salience: EndingSalience = Field(
        description=(
            "How much this dilemma drives ending differentiation. "
            "'high' = endings MUST differ; 'low' = endings MAY acknowledge; "
            "'none' = ending prose MUST NOT reference this choice"
        ),
    )
    residue_weight: ResidueWeight = Field(
        description=(
            "How much mid-story prose varies based on this dilemma's outcome. "
            "'heavy' = shared passages MUST show state-specific differences; "
            "'light' = shared passages MAY acknowledge; "
            "'cosmetic' = shared passages MUST NOT reference this choice"
        ),
    )
    payoff_budget: int = Field(
        ge=2,
        le=6,
        description="Minimum exclusive beats before convergence",
    )
    reasoning: str = Field(
        min_length=10,
        max_length=300,
        description="Chain-of-thought for the classification",
    )
    convergence_point: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Where this dilemma's paths converge (location-based, concrete)",
    )
    residue_note: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Differences that persist after convergence for this dilemma",
    )
    ending_tone: str | None = Field(
        default=None,
        min_length=2,
        max_length=80,
        description=(
            "Emotional quality of endings shaped by this dilemma's resolution "
            "(2-5 words, e.g. 'bittersweet triumph', 'cold justice')"
        ),
    )


class DilemmaRelationship(BaseModel):
    """Pairwise dilemma ordering relationship from SEED (Document 3, Part 2).

    Serialized as Section 8 (after prune). Tells GROW how two dilemmas
    relate temporally: one wrapping the other, running concurrently, or
    sequentially.

    For ``concurrent`` (symmetric) ordering, the pair is normalized to
    canonical order (dilemma_a < dilemma_b) to prevent A-B / B-A duplicates.
    For directional orderings (``wraps``, ``serial``), the supplied order is
    preserved because direction matters: "A wraps B" != "B wraps A".

    Note: ``shared_entity`` is NOT an ordering relationship — it is derived
    from ``anchored_to`` edges in the graph.

    Attributes:
        dilemma_a: First dilemma ID. For ``concurrent``, lexicographically smaller after normalization.
        dilemma_b: Second dilemma ID. For ``concurrent``, lexicographically larger after normalization.
        ordering: Temporal ordering between the dilemmas (wraps|concurrent|serial).
        description: What the relationship means narratively.
        reasoning: Chain-of-thought for the classification.
    """

    dilemma_a: str = Field(min_length=1, description="First dilemma ID")
    dilemma_b: str = Field(min_length=1, description="Second dilemma ID")
    ordering: DilemmaOrdering = Field(
        description="Temporal ordering (wraps|concurrent|serial)",
    )
    description: str = Field(
        min_length=1, max_length=500, description="Narrative meaning of the relationship"
    )
    reasoning: str = Field(
        min_length=10,
        max_length=300,
        description="Chain-of-thought for the classification",
    )

    @model_validator(mode="after")
    def _validate_and_normalize_pair(self) -> DilemmaRelationship:
        """Reject self-referential pairs; normalize symmetric orderings to canonical order.

        Only ``concurrent`` pairs are alphabetically normalized (a < b) because the
        relationship is symmetric.  ``wraps`` and ``serial`` are directional — swapping
        would invert meaning ("A wraps B" becomes "B wraps A").
        """
        if self.dilemma_a == self.dilemma_b:
            msg = f"dilemma_a and dilemma_b cannot be the same: {self.dilemma_a}"
            raise ValueError(msg)
        if self.ordering == "concurrent" and self.dilemma_a > self.dilemma_b:
            self.dilemma_a, self.dilemma_b = self.dilemma_b, self.dilemma_a
        return self

    @property
    def pair_key(self) -> str:
        """Composite dedup key for the (dilemma_a, dilemma_b) pair."""
        return f"{self.dilemma_a}__{self.dilemma_b}"


class SeedOutput(BaseModel):
    """Complete output of the SEED stage.

    SEED transforms brainstorm material into committed story structure.
    After SEED, no new paths can be created (PATH FREEZE).

    Attributes:
        entities: Entity curation decisions.
        dilemmas: Dilemma exploration decisions.
        paths: Created plot paths.
        consequences: Narrative consequences for paths.
        initial_beats: Initial beats for each path.
        dilemma_analyses: Per-dilemma convergence classifications (Section 7).
        dilemma_relationships: Pairwise dilemma ordering relationships (Section 8).
    """

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )
    dilemmas: list[DilemmaDecision] = Field(
        default_factory=list,
        description="Dilemma exploration decisions",
    )
    paths: list[Path] = Field(
        default_factory=list,
        description="Created plot paths",
    )
    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for paths",
    )
    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each path",
    )
    dilemma_analyses: list[DilemmaAnalysis] = Field(
        default_factory=list,
        description="Per-dilemma convergence classifications (Section 7, post-prune)",
    )
    dilemma_relationships: list[DilemmaRelationship] = Field(
        default_factory=list,
        description="Pairwise dilemma ordering relationships (Section 8, post-prune)",
    )


# Section wrapper models for iterative serialization
# These allow serializing SEED output in chunks to avoid output truncation


def _deduplicate_and_check(items: list[Any], id_attr: str, type_name: str) -> list[Any]:
    """Silently drop identical duplicates; raise on non-identical ones.

    LLMs sometimes emit the same item twice verbatim. Wasting a retry loop
    on that is pointless, so identical copies are silently dropped (keeping
    the first occurrence). Non-identical duplicates (same ID, different
    content) are real conflicts and still raise ValueError.

    Returns the deduplicated list.
    """
    seen: dict[str, Any] = {}
    result: list[Any] = []
    for item in items:
        item_id = getattr(item, id_attr)
        if item_id not in seen:
            seen[item_id] = item
            result.append(item)
        elif item == seen[item_id]:
            # Identical duplicate — silently drop
            continue
        else:
            # Same ID, different content — keep so the check below catches it
            result.append(item)

    ids = [getattr(item, id_attr) for item in result]
    dupes = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if dupes:
        msg = (
            f"Duplicates found for {type_name}: {sorted(dupes)}. "
            f"REMOVE the duplicate entries so that each item has a unique {type_name}. "
            f"Do NOT rename duplicates — delete the extra copy."
        )
        raise ValueError(msg)
    return result


class EntitiesSection(BaseModel):
    """Wrapper for serializing entity decisions separately."""

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )

    @model_validator(mode="after")
    def _deduplicate_entities(self) -> EntitiesSection:
        """Drop identical duplicate entities; raise on conflicting ones."""
        self.entities = _deduplicate_and_check(self.entities, "entity_id", "entity_id")
        return self


class DilemmasSection(BaseModel):
    """Wrapper for serializing dilemma decisions separately."""

    dilemmas: list[DilemmaDecision] = Field(
        default_factory=list,
        description="Dilemma exploration decisions",
    )

    @model_validator(mode="after")
    def _deduplicate_dilemmas(self) -> DilemmasSection:
        """Drop identical duplicate dilemmas; raise on conflicting ones."""
        self.dilemmas = _deduplicate_and_check(self.dilemmas, "dilemma_id", "dilemma_id")
        return self


class PathsSection(BaseModel):
    """Wrapper for serializing paths separately."""

    paths: list[Path] = Field(
        default_factory=list,
        description="Created plot paths",
    )

    @model_validator(mode="after")
    def _deduplicate_paths(self) -> PathsSection:
        """Drop identical duplicate paths; raise on conflicting ones."""
        self.paths = _deduplicate_and_check(self.paths, "path_id", "path_id")
        return self


class DilemmaPathsSection(BaseModel):
    """Wrapper for serializing paths for a single dilemma.

    Used by per-dilemma path serialization to constrain the LLM to generating
    paths for exactly one dilemma. This prevents trailing dilemmas from being
    dropped when serializing all paths at once.
    """

    paths: list[Path] = Field(
        min_length=1,
        max_length=4,
        description="1-4 paths for this specific dilemma",
    )

    @model_validator(mode="after")
    def _deduplicate_paths(self) -> DilemmaPathsSection:
        """Drop identical duplicate paths; raise on conflicting ones."""
        self.paths = _deduplicate_and_check(self.paths, "path_id", "path_id")
        return self


class ConsequencesSection(BaseModel):
    """Wrapper for serializing consequences separately."""

    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for paths",
    )

    @model_validator(mode="after")
    def _deduplicate_consequences(self) -> ConsequencesSection:
        """Drop identical duplicate consequences; raise on conflicting ones."""
        self.consequences = _deduplicate_and_check(
            self.consequences, "consequence_id", "consequence_id"
        )
        return self


class BeatsSection(BaseModel):
    """Wrapper for serializing initial beats separately."""

    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each path",
    )


class PathBeatsSection(BaseModel):
    """Wrapper for serializing beats for a single path.

    Used by per-path beat serialization to constrain the LLM to generating
    beats for exactly one path with a fixed dilemma_id. This makes the
    path→dilemma alignment trivial since the context only contains one valid
    dilemma for dilemma_impacts.
    """

    initial_beats: list[InitialBeat] = Field(
        min_length=2,
        max_length=4,
        description="2-4 initial beats for this specific path",
    )

    @model_validator(mode="after")
    def _deduplicate_beats(self) -> PathBeatsSection:
        """Drop identical duplicate beats; raise on conflicting ones."""
        self.initial_beats = _deduplicate_and_check(self.initial_beats, "beat_id", "beat_id")
        return self


class DilemmaAnalysisSection(BaseModel):
    """Wrapper for serializing dilemma analyses separately (Section 7)."""

    dilemma_analyses: list[DilemmaAnalysis] = Field(
        default_factory=list,
        description="Per-dilemma convergence classifications",
    )

    @model_validator(mode="after")
    def _deduplicate_analyses(self) -> DilemmaAnalysisSection:
        """Drop identical duplicate analyses; raise on conflicting ones."""
        self.dilemma_analyses = _deduplicate_and_check(
            self.dilemma_analyses, "dilemma_id", "dilemma_id"
        )
        return self


class DilemmaRelationshipsSection(BaseModel):
    """Wrapper for serializing dilemma ordering relationships separately (Section 8)."""

    dilemma_relationships: list[DilemmaRelationship] = Field(
        default_factory=list,
        description="Pairwise dilemma ordering relationships",
    )

    @model_validator(mode="after")
    def _deduplicate_relationships(self) -> DilemmaRelationshipsSection:
        """Drop identical duplicate relationships; raise on conflicting ones."""
        self.dilemma_relationships = _deduplicate_and_check(
            self.dilemma_relationships, "pair_key", "pair_key"
        )
        return self


def make_constrained_dilemmas_section(
    answer_ids_by_dilemma: dict[str, list[str]],
) -> type[BaseModel]:
    """Create a DilemmasSection with enum-constrained answer and dilemma IDs.

    Builds dynamic Pydantic models where ``dilemma_id``, ``explored``, and
    ``unexplored`` fields use StrEnum values derived from the brainstorm graph.
    The resulting JSON schema contains ``enum`` constraints that prevent the LLM
    from emitting invalid IDs at the token level (constrained decoding in
    llama.cpp / Ollama).

    The enum covers *all* answer IDs across all dilemmas (a global pool).
    Per-dilemma validation (answer X belongs to dilemma Y) is handled by
    ``_early_validate_dilemma_answers`` in the serialize layer.

    Args:
        answer_ids_by_dilemma: Mapping of raw dilemma IDs to their valid
            raw answer IDs (as returned by ``get_brainstorm_answer_ids``).

    Returns:
        A BaseModel subclass structurally identical to ``DilemmasSection``
        but with enum-constrained ID fields.  Falls back to the unconstrained
        ``DilemmasSection`` when the input is empty.
    """
    # Filter out dilemmas with no answers (defensive — get_brainstorm_answer_ids
    # only includes dilemmas that have answers, but callers may pass other data).
    valid_dilemmas = {d: a for d, a in answer_ids_by_dilemma.items() if a}
    if not valid_dilemmas:
        return DilemmasSection

    all_answers = sorted({aid for aids in valid_dilemmas.values() for aid in aids})
    all_dilemma_ids = sorted(valid_dilemmas)

    # Build StrEnum types — these surface as JSON schema ``enum`` arrays.
    # mypy cannot statically determine members from dict comprehensions,
    # but the enums are correct at runtime.
    AnswerIdEnum = StrEnum("AnswerIdEnum", {a: a for a in all_answers})  # type: ignore[misc]
    # Dilemma IDs are scoped (e.g., "dilemma::trust_or_betray") per v5 conventions.
    # Answer IDs remain unscoped (e.g., "trust") as they're local to their dilemma.
    DilemmaIdEnum = StrEnum(  # type: ignore[misc]
        "DilemmaIdEnum",
        {d: f"dilemma::{d}" for d in all_dilemma_ids},
    )

    # Use inheritance so validators (migrate_considered_field, _deduplicate_dilemmas)
    # are inherited from the base classes rather than duplicated.
    class ConstrainedDilemmaDecision(DilemmaDecision):
        """DilemmaDecision with enum-constrained IDs."""

        dilemma_id: DilemmaIdEnum = Field(
            description="Dilemma ID from BRAINSTORM",
        )
        explored: list[AnswerIdEnum] = Field(  # type: ignore[assignment]
            min_length=1,
            description="Answer IDs the LLM intended to explore as paths",
        )
        unexplored: list[AnswerIdEnum] = Field(  # type: ignore[assignment]
            default_factory=list,
            description="Answer IDs not explored (become shadows)",
        )

    class ConstrainedDilemmasSection(DilemmasSection):
        """DilemmasSection with enum-constrained IDs."""

        dilemmas: list[ConstrainedDilemmaDecision] = Field(  # type: ignore[assignment]
            default_factory=list,
            description="Dilemma exploration decisions",
        )

    return ConstrainedDilemmasSection
