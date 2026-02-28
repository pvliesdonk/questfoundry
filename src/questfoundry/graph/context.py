"""Graph context formatting for LLM prompts.

Provides functions to format graph data as context for LLM serialization,
giving the model authoritative lists of valid IDs to reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.grow_context import format_grow_valid_ids

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph
    from questfoundry.models.seed import Consequence, Path, SeedOutput

# Entity categories - used as scope prefixes for entity IDs
# Format: category::name (e.g., character::pim, location::manor)
ENTITY_CATEGORIES = frozenset(["character", "location", "object", "faction"])

# Scope prefixes for typed IDs (disambiguates ID types for LLM)
# Note: Entities use category as prefix (character::, location::, etc.)
# rather than a generic "entity::" prefix
SCOPE_DILEMMA = "dilemma"
SCOPE_PATH = "path"

# Development state names (computed, not stored)
STATE_COMMITTED = "committed"
STATE_DEFERRED = "deferred"
STATE_LATENT = "latent"


def count_paths_per_dilemma(seed_output: SeedOutput) -> dict[str, int]:
    """Count how many paths exist for each dilemma.

    This is the authoritative source for path existence - used to derive
    development state (committed vs deferred) and compute arc counts.

    Args:
        seed_output: The SEED output containing paths.

    Returns:
        Dict mapping dilemma_id (raw, unscoped) to count of paths.
    """
    counts: dict[str, int] = {}
    for path in seed_output.paths:
        did = path.dilemma_id
        # Handle scoped IDs (dilemma::foo -> foo)
        if "::" in did:
            did = did.split("::", 1)[1]
        counts[did] = counts.get(did, 0) + 1
    return counts


def get_dilemma_development_states(
    seed_output: SeedOutput,
) -> dict[str, dict[str, str]]:
    """Compute development states for all answers in a SEED output.

    Development states are derived from comparing the `explored` field
    (LLM intent) against actual path existence:
    - **committed**: Answer in `explored` AND has a path
    - **deferred**: Answer in `explored` but NO path (pruned)
    - **latent**: Answer not in `explored` (never intended for exploration)

    Args:
        seed_output: The SEED output containing dilemma decisions and paths.

    Returns:
        Dict mapping dilemma_id to dict mapping answer_id to state.
        Example: {"mentor_trust": {"protector": "committed", "manipulator": "deferred"}}
    """
    # Build lookup of which answers have paths
    # path.answer_id is the raw local ID (not prefixed)
    answer_has_path: dict[str, set[str]] = {}
    for path in seed_output.paths:
        did = path.dilemma_id
        # Handle scoped IDs
        if "::" in did:
            did = did.split("::", 1)[1]
        if did not in answer_has_path:
            answer_has_path[did] = set()
        answer_has_path[did].add(path.answer_id)

    # Compute states for each dilemma's answers
    states: dict[str, dict[str, str]] = {}
    for dilemma in seed_output.dilemmas:
        did = dilemma.dilemma_id
        # Handle scoped IDs
        if "::" in did:
            did = did.split("::", 1)[1]

        dilemma_states: dict[str, str] = {}
        explored_set = set(dilemma.explored)
        unexplored_set = set(dilemma.unexplored)
        path_answers = answer_has_path.get(did, set())

        # Process all known answers (from explored + unexplored)
        all_answers = explored_set | unexplored_set
        for ans_id in all_answers:
            if ans_id in explored_set:
                if ans_id in path_answers:
                    dilemma_states[ans_id] = STATE_COMMITTED
                else:
                    dilemma_states[ans_id] = STATE_DEFERRED
            else:
                # In unexplored, never explored
                dilemma_states[ans_id] = STATE_LATENT

        states[did] = dilemma_states

    return states


def parse_scoped_id(scoped_id: str) -> tuple[str, str]:
    """Parse 'type::raw_id' into (type, raw_id).

    Scoped IDs use '::' as delimiter to disambiguate ID types for the LLM.
    This prevents confusion between entity IDs, dilemma IDs, and path IDs.

    Args:
        scoped_id: An ID string, optionally scoped (e.g., 'entity::hero' or 'hero').

    Returns:
        Tuple of (scope_type, raw_id). Returns ('', raw_id) if unscoped.

    Examples:
        >>> parse_scoped_id("entity::hero")
        ('entity', 'hero')
        >>> parse_scoped_id("path::host_motive")
        ('path', 'host_motive')
        >>> parse_scoped_id("hero")
        ('', 'hero')
    """
    if "::" in scoped_id:
        parts = scoped_id.split("::", 1)
        return parts[0], parts[1]
    return "", scoped_id


def strip_scope_prefix(scoped_id: str) -> str:
    """Strip scope prefix from ID, returning only the raw ID.

    This is a convenience wrapper around parse_scoped_id() for the common
    case where you only need the raw ID, not the scope.

    Args:
        scoped_id: An ID string, optionally scoped (e.g., 'entity::hero').

    Returns:
        Raw ID without scope prefix (e.g., 'hero').

    Examples:
        >>> strip_scope_prefix("entity::hero")
        'hero'
        >>> strip_scope_prefix("path::host_motive")
        'host_motive'
        >>> strip_scope_prefix("hero")
        'hero'
    """
    _, raw_id = parse_scoped_id(scoped_id)
    return raw_id


def get_default_answer_from_graph(graph: Graph, dilemma_id: str) -> str | None:
    """Look up the default (canonical) answer for a dilemma from the graph.

    Uses the ``is_canonical`` flag on answer nodes rather than relying on
    the ordering of the ``explored`` list, which LLMs do not guarantee.

    Args:
        graph: The story graph containing dilemma and answer nodes.
        dilemma_id: Raw or scoped dilemma ID.

    Returns:
        The raw answer ID marked as default, or None if not found.
    """
    raw_did = strip_scope_prefix(dilemma_id)
    prefixed_did = f"dilemma::{raw_did}"
    alt_edges = graph.get_edges(from_id=prefixed_did, edge_type="has_answer")
    for edge in alt_edges:
        alt_node = graph.get_node(edge["to"])
        if alt_node and alt_node.get("is_canonical"):
            raw_id = alt_node.get("raw_id")
            if raw_id:
                return str(raw_id)
    return None


def format_scoped_id(scope: str, raw_id: str) -> str:
    """Format a raw ID with its scope prefix.

    Args:
        scope: The scope type (e.g., 'entity', 'dilemma', 'path').
        raw_id: The raw ID without scope.

    Returns:
        Scoped ID string (e.g., 'entity::hero').
    """
    return f"{scope}::{raw_id}"


def normalize_scoped_id(raw_id: str, scope: str) -> str:
    """Ensure an ID has the scope prefix, adding it if missing.

    Args:
        raw_id: ID that may or may not already have the scope prefix.
        scope: The scope type (e.g., 'dilemma', 'path').

    Returns:
        ID with scope prefix guaranteed (e.g., 'dilemma::mentor_trust').
    """
    prefix = f"{scope}::"
    return raw_id if raw_id.startswith(prefix) else f"{prefix}{raw_id}"


def get_passage_beats(graph: Graph, passage_id: str) -> list[str]:
    """Get beat IDs grouped into a passage via ``grouped_in`` edges.

    Args:
        graph: The story graph.
        passage_id: Passage node ID to look up.

    Returns:
        List of beat IDs (typically single-element, multiple for merged passages).
    """
    edges = graph.get_edges(to_id=passage_id, edge_type="grouped_in")
    return sorted(e["from"] for e in edges)


def get_primary_beat(graph: Graph, passage_id: str) -> str | None:
    """Get the primary beat ID for a passage.

    Returns the first beat from the ``grouped_in`` edges.

    Args:
        graph: The story graph.
        passage_id: Passage node ID to look up.

    Returns:
        Primary beat ID, or None if passage has no beats.
    """
    beats = get_passage_beats(graph, passage_id)
    return beats[0] if beats else None


def is_entity_id(scoped_id: str) -> bool:
    """Check if an ID has an entity category prefix.

    Entity IDs use category as their prefix (character::, location::, etc.)
    rather than a generic "entity::" prefix.

    Args:
        scoped_id: An ID to check.

    Returns:
        True if the ID starts with a valid entity category prefix.

    Examples:
        >>> is_entity_id("character::pim")
        True
        >>> is_entity_id("location::manor")
        True
        >>> is_entity_id("dilemma::trust")
        False
        >>> is_entity_id("pim")
        False
    """
    scope, _ = parse_scoped_id(scoped_id)
    return scope in ENTITY_CATEGORIES


def format_entity_id(category: str, raw_id: str) -> str:
    """Create a category-prefixed entity ID.

    Entity IDs use the category as their prefix for semantic clarity:
    - character::pim (not entity::char_pim)
    - location::manor (not entity::loc_manor)

    Args:
        category: Entity category (character, location, object, faction).
        raw_id: Raw entity name without prefix.

    Returns:
        Category-prefixed entity ID.

    Raises:
        ValueError: If category is not a valid entity category.

    Examples:
        >>> format_entity_id("character", "pim")
        'character::pim'
        >>> format_entity_id("location", "manor")
        'location::manor'
    """
    if category not in ENTITY_CATEGORIES:
        raise ValueError(
            f"Invalid entity category '{category}'. "
            f"Must be one of: {', '.join(sorted(ENTITY_CATEGORIES))}"
        )
    # Strip any existing prefix from raw_id
    if "::" in raw_id:
        raw_id = raw_id.rsplit("::", 1)[-1]
    return f"{category}::{raw_id}"


def parse_entity_id(entity_id: str) -> tuple[str, str]:
    """Parse a category-prefixed entity ID into (category, raw_id).

    Args:
        entity_id: An entity ID (e.g., 'character::pim').

    Returns:
        Tuple of (category, raw_id).

    Raises:
        ValueError: If entity_id doesn't have a valid category prefix.

    Examples:
        >>> parse_entity_id("character::pim")
        ('character', 'pim')
        >>> parse_entity_id("location::manor")
        ('location', 'manor')
    """
    scope, raw_id = parse_scoped_id(entity_id)
    if scope not in ENTITY_CATEGORIES:
        raise ValueError(
            f"Entity ID '{entity_id}' has invalid category prefix '{scope}'. "
            f"Must be one of: {', '.join(sorted(ENTITY_CATEGORIES))}"
        )
    return scope, raw_id


def parse_hierarchical_path_id(path_id: str) -> tuple[str, str]:
    """Extract dilemma_id and answer_id from a hierarchical path ID.

    Hierarchical path IDs embed the parent dilemma in the ID itself:
    `path::dilemma_id__answer_id`

    This format makes the path→dilemma relationship explicit and prevents
    LLM misreferences.

    Args:
        path_id: A path ID string (e.g., 'path::mentor_trust__benevolent').

    Returns:
        Tuple of (dilemma_id, answer_id) where dilemma_id includes
        the 'dilemma::' prefix.

    Raises:
        ValueError: If path_id doesn't contain '__' separator.

    Examples:
        >>> parse_hierarchical_path_id("path::mentor_trust__benevolent")
        ('dilemma::mentor_trust', 'benevolent')
        >>> parse_hierarchical_path_id("path::mentor_trust__selfish")
        ('dilemma::mentor_trust', 'selfish')
    """
    scope, raw = parse_scoped_id(path_id)
    if scope not in ("", SCOPE_PATH):
        raise ValueError(f"Path ID '{path_id}' has wrong scope prefix (expected '{SCOPE_PATH}::')")
    if "__" not in raw:
        raise ValueError(f"Path ID '{path_id}' is not hierarchical (missing '__' separator)")
    dilemma_raw, answer_id = raw.rsplit("__", 1)
    return f"{SCOPE_DILEMMA}::{dilemma_raw}", answer_id


def format_hierarchical_path_id(dilemma_id: str, answer_id: str) -> str:
    """Create a hierarchical path ID from dilemma and answer components.

    Creates path IDs in the format `path::dilemma_id__answer_id` which
    embeds the parent dilemma relationship in the ID itself.

    Args:
        dilemma_id: The parent dilemma ID (with or without 'dilemma::' prefix).
        answer_id: The answer ID this path explores.

    Returns:
        Hierarchical path ID string (e.g., 'path::mentor_trust__benevolent').

    Examples:
        >>> format_hierarchical_path_id("dilemma::mentor_trust", "benevolent")
        'path::mentor_trust__benevolent'
        >>> format_hierarchical_path_id("mentor_trust", "selfish")
        'path::mentor_trust__selfish'
    """
    dilemma_raw = strip_scope_prefix(dilemma_id)  # "mentor_trust"
    return f"{SCOPE_PATH}::{dilemma_raw}__{answer_id}"


def format_answer_ids_by_dilemma(dilemmas: list[dict[str, Any]]) -> str:
    """Format answer IDs per dilemma as context for paths serialization.

    After dilemmas are serialized, each dilemma has ``explored`` and
    ``unexplored`` answer lists.  Injecting these before the paths section
    lets the model know exactly which answer_id values are valid for each
    dilemma, preventing phantom answer references.

    Args:
        dilemmas: List of dilemma decision dicts from serialized output.

    Returns:
        Formatted manifest string, or empty string if no dilemmas.
    """
    if not dilemmas:
        return ""

    dilemma_lines = []
    for d in sorted(dilemmas, key=lambda x: x.get("dilemma_id", "")):
        dilemma_id = d.get("dilemma_id", "")
        if not dilemma_id:
            continue
        scoped = normalize_scoped_id(strip_scope_prefix(dilemma_id), SCOPE_DILEMMA)
        explored = d.get("explored", [])
        unexplored = d.get("unexplored", [])
        dilemma_lines.append(f"- `{scoped}` -> explored: {explored}, unexplored: {unexplored}")

    if not dilemma_lines:
        return ""

    lines = [
        "## Valid Answer IDs per Dilemma",
        "",
        "Each path's `answer_id` MUST be one of the `explored` IDs below.",
        "Do NOT invent answer IDs or use `unexplored` IDs as path answer_ids.",
        "",
        *dilemma_lines,
        "",
    ]
    return "\n".join(lines)


def format_path_ids_context(paths: list[dict[str, Any]]) -> str:
    """Format path IDs for beat serialization with inline constraints.

    Includes path→dilemma mapping so the model knows which dilemma_id
    to use in dilemma_impacts for each beat's path.

    Small models don't "look back" at referenced sections, so we
    embed constraints at the point of use.

    Args:
        paths: List of path dicts from serialized PathsSection.

    Returns:
        Formatted context string with path IDs, or empty string if none.
    """
    if not paths:
        return ""

    # Sort for deterministic output across runs.
    # Seed output should use raw IDs (e.g., "mentor_trust__protector"); tolerate scoped "path::..."
    # without accidentally normalizing other legacy shorthand scopes.
    def _raw_path_id(pid: str) -> str:
        return pid.split("::", 1)[1] if pid.startswith(f"{SCOPE_PATH}::") else pid

    path_ids = sorted(_raw_path_id(pid) for p in paths if (pid := p.get("path_id")))

    if not path_ids:
        return ""

    # Pipe-delimited for easy scanning, with path:: scope prefix
    id_list = " | ".join(f"`{normalize_scoped_id(pid, SCOPE_PATH)}`" for pid in path_ids)

    lines = [
        "## VALID PATH IDs (copy exactly, no modifications)",
        "",
        f"Allowed: {id_list}",
        "",
    ]

    # Build path→dilemma mapping for dilemma_impacts guidance
    path_dilemma_pairs = []
    for p in sorted(paths, key=lambda x: x.get("path_id", "")):
        pid = p.get("path_id")
        pid_raw = _raw_path_id(pid) if pid else None
        dilemma_id = p.get("dilemma_id", "")
        if pid and dilemma_id:
            # Strip scope prefix if present for clean display
            raw_dilemma = dilemma_id.split("::", 1)[-1] if "::" in dilemma_id else dilemma_id
            if pid_raw:
                path_dilemma_pairs.append((pid_raw, raw_dilemma))

    if path_dilemma_pairs:
        lines.append("## PATH → DILEMMA MAPPING (use in dilemma_impacts)")
        lines.append("Each beat's dilemma_impacts should use its path's parent dilemma:")
        lines.append("")
        for pid, dilemma in path_dilemma_pairs:
            lines.append(
                f"  - `{normalize_scoped_id(pid, SCOPE_PATH)}` → "
                f"`{normalize_scoped_id(dilemma, SCOPE_DILEMMA)}`"
            )
        lines.append("")

    lines.extend(
        [
            "Rules:",
            "- Use ONLY IDs from the list above in the `paths` array",
            "- Include the `path::` prefix in your output",
            "- For dilemma_impacts, use the dilemma from the mapping above",
            "- Do NOT derive IDs from dilemma concepts",
            "- Do NOT use entity IDs as dilemma_ids",
            "- If a concept has no matching path, omit it - do NOT invent an ID",
            "",
            "WRONG (will fail validation):",
            "- `clock_distortion` - NOT a valid path ID (derived from concept)",
            "- `host_motive` - WRONG: missing scope prefix, use `path::host_motive`",
            "- `dilemma::seed_of_stillness` - WRONG: entity ID used as dilemma",
            "",
        ]
    )

    return "\n".join(lines)


def format_valid_ids_context(graph: Graph, stage: str, section: str | None = None) -> str:
    """Format valid IDs as context for LLM serialization.

    Provides the authoritative list of IDs the LLM must use.
    This prevents phantom ID references by showing valid options upfront.

    Args:
        graph: Graph containing nodes from previous stages.
        stage: Current stage name ("seed", "grow", etc.).
        section: When set, scope output to only the IDs relevant to this
            serialization section (e.g. "entities", "dilemmas").  ``None``
            returns the full manifest (monolithic mode / backward compat).

    Returns:
        Formatted context string, or empty string if not applicable.
    """
    if stage == "seed":
        return _format_seed_valid_ids(graph, section=section)
    if stage == "grow":
        return _format_grow_valid_ids(graph)
    return ""


def _format_grow_valid_ids(graph: Graph) -> str:
    """Format GROW-stage valid IDs for LLM serialization.

    Delegates to grow_context module and formats the result as a
    human-readable context string with all ID types.

    Args:
        graph: Graph containing SEED/GROW data.

    Returns:
        Formatted context string with valid IDs.
    """
    ids = format_grow_valid_ids(graph)
    lines = ["## VALID IDs FOR GROW PHASES", ""]

    for label, key in [
        ("Beat IDs", "valid_beat_ids"),
        ("Path IDs", "valid_path_ids"),
        ("Dilemma IDs", "valid_dilemma_ids"),
        ("Entity IDs", "valid_entity_ids"),
        ("Passage IDs", "valid_passage_ids"),
        ("Choice IDs", "valid_choice_ids"),
    ]:
        value = ids.get(key, "")
        if value:
            lines.append(f"### {label}")
            lines.append(value)
            lines.append("")

    return "\n".join(lines) if any(ids.values()) else ""


def _format_seed_valid_ids(graph: Graph, section: str | None = None) -> str:
    """Format BRAINSTORM IDs for SEED serialization.

    Groups entities by category and lists dilemmas with their answers,
    making it clear which IDs are valid for the SEED stage to reference.
    Includes counts to enable completeness validation.

    When *section* is ``None`` the full manifest is returned (monolithic mode).
    When *section* is ``"entities"`` only entity IDs are included; when
    ``"dilemmas"`` only dilemma IDs.  Any other value returns ``""``.

    Args:
        graph: Graph containing BRAINSTORM data.
        section: Serialization section to scope output for, or ``None``
            for the full manifest.

    Returns:
        Formatted context string with valid IDs and counts.
    """
    include_entities = section is None or section == "entities"
    include_dilemmas = section is None or section == "dilemmas"

    if not include_entities and not include_dilemmas:
        return ""

    # Get expected counts from canonical source (DRY principle)
    counts = get_expected_counts(graph)
    total_entity_count = counts["entities"]
    dilemma_count = counts["dilemmas"]

    # Header — scoped to what this section generates
    if section is None:
        lines = [
            "## VALID IDS MANIFEST - GENERATE FOR ALL",
            "",
            "You MUST generate a decision for EVERY ID listed below.",
            "Missing items WILL cause validation failure.",
            "",
        ]
    elif section == "entities":
        lines = [
            "## VALID ENTITY IDS - GENERATE FOR ALL",
            "",
            "You MUST generate a decision for EVERY entity ID listed below.",
            "Missing items WILL cause validation failure.",
            "",
        ]
    elif section == "dilemmas":
        lines = [
            "## VALID DILEMMA IDS - GENERATE FOR ALL",
            "",
            "You MUST generate a decision for EVERY dilemma ID listed below.",
            "Missing items WILL cause validation failure.",
            "",
        ]

    # Entity IDs block
    if include_entities:
        # Group entities by type (character, location, object, faction)
        # Track which entities need names (don't have one from BRAINSTORM)
        entities = graph.get_nodes_by_type("entity")
        by_category: dict[str, list[tuple[str, bool]]] = {}  # (raw_id, needs_name)
        for node in entities.values():
            cat = node.get("entity_type", "unknown")
            raw_id = node.get("raw_id", "")
            if raw_id:  # Only include entities with valid raw_id
                # Check if entity has a name from BRAINSTORM
                needs_name = not node.get("name")
                by_category.setdefault(cat, []).append((raw_id, needs_name))

        if by_category:
            lines.append(
                f"### Entity IDs (TOTAL: {total_entity_count} - generate decision for ALL)"
            )
            lines.append("Use these for `entity_id`, `entities`, and `location` fields:")
            lines.append("Entities marked (needs name) require a `name` field if RETAINED.")
            lines.append("")

            for category in ENTITY_CATEGORIES:
                if category in by_category:
                    cat_count = len(by_category[category])
                    lines.append(f"**{category.title()}s ({cat_count}):**")
                    for raw_id, needs_name in sorted(by_category[category]):
                        marker = " (needs name)" if needs_name else ""
                        lines.append(f"  - `{category}::{raw_id}`{marker}")
                    lines.append("")

    # Dilemma IDs block
    if include_dilemmas:
        dilemmas = graph.get_nodes_by_type("dilemma")
        if dilemmas:
            # Pre-build answer edges map to avoid O(D*E) lookups
            answer_edges_by_dilemma: dict[str, list[dict[str, Any]]] = {}
            for edge in graph.get_edges(edge_type="has_answer"):
                from_id = edge.get("from")
                if from_id:
                    answer_edges_by_dilemma.setdefault(from_id, []).append(edge)

            lines.append(f"### Dilemma IDs (TOTAL: {dilemma_count} - generate decision for ALL)")
            lines.append("Format: dilemma::id → [answer_ids]")
            lines.append("Note: Every dilemma_id contains `_or_` — entity IDs never do.")
            lines.append("")

            for did, ddata in sorted(dilemmas.items()):
                raw_id = ddata.get("raw_id")
                if not raw_id:
                    continue

                answers = []
                for edge in answer_edges_by_dilemma.get(did, []):
                    answer_node = graph.get_node(edge.get("to", ""))
                    if answer_node:
                        ans_id = answer_node.get("raw_id")
                        if ans_id:
                            default = " (default)" if answer_node.get("is_canonical") else ""
                            answers.append(f"`{ans_id}`{default}")

                if answers:
                    # Sort answers for deterministic output
                    answers.sort()
                    lines.append(
                        f"- `{normalize_scoped_id(raw_id, SCOPE_DILEMMA)}` → [{', '.join(answers)}]"
                    )

            lines.append("")

    # Generation requirements — scoped to included ID types
    _append_seed_requirements(lines, include_entities, include_dilemmas, counts)

    return "\n".join(lines)


def _append_seed_requirements(
    lines: list[str],
    include_entities: bool,
    include_dilemmas: bool,
    counts: dict[str, int],
) -> None:
    """Append generation requirements scoped to the included ID types."""
    total_entity_count = counts["entities"]
    dilemma_count = counts["dilemmas"]
    entity_word = "item" if total_entity_count == 1 else "items"
    dilemma_word = "item" if dilemma_count == 1 else "items"

    lines.append("### Generation Requirements (CRITICAL)")

    if include_entities:
        lines.append(
            f"- Generate EXACTLY {total_entity_count} entity decisions (one per entity above)"
        )
        lines.append(
            "- Use scoped IDs: category prefix for entities (character::, location::, etc.)"
        )
    if include_dilemmas:
        lines.append(
            f"- Generate EXACTLY {dilemma_count} dilemma decisions (one per dilemma above)"
        )
        lines.append("- Path `answer_id` must be from that dilemma's answers list")
        lines.append("- Use scoped IDs: dilemma:: for dilemmas")

    lines.append("")
    lines.append("### Verification")
    lines.append("Before submitting, COUNT your outputs:")

    if include_entities:
        lines.append(f"- entities array should have {total_entity_count} {entity_word}")
    if include_dilemmas:
        lines.append(f"- dilemmas array should have {dilemma_count} {dilemma_word}")


def get_expected_counts(graph: Graph) -> dict[str, int]:
    """Get expected counts for SEED output validation.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Dict with expected counts for each section.
    """
    entities = graph.get_nodes_by_type("entity")
    dilemmas = graph.get_nodes_by_type("dilemma")

    entity_count = sum(1 for node in entities.values() if node.get("raw_id"))
    dilemma_count = sum(1 for ddata in dilemmas.values() if ddata.get("raw_id"))

    return {
        "entities": entity_count,
        "dilemmas": dilemma_count,
    }


def get_brainstorm_answer_ids(graph: Graph) -> dict[str, list[str]]:
    """Get authoritative answer IDs per dilemma from the brainstorm graph.

    Returns the ground-truth answer IDs as defined during BRAINSTORM,
    not the LLM's serialized output. Used for early validation after
    dilemma serialization to catch answer ID mismatches before they
    cascade to path generation.

    Args:
        graph: Graph containing BRAINSTORM dilemma and answer nodes.

    Returns:
        Dict mapping raw dilemma ID to list of raw answer IDs.
        Example: {"trust_or_betray": ["trust", "betray"]}
    """
    dilemmas = graph.get_nodes_by_type("dilemma")
    if not dilemmas:
        return {}

    # Pre-build answer edges map
    answer_edges_by_dilemma: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get_edges(edge_type="has_answer"):
        from_id = edge.get("from")
        if from_id:
            answer_edges_by_dilemma.setdefault(from_id, []).append(edge)

    result: dict[str, list[str]] = {}
    for did, ddata in dilemmas.items():
        raw_id = ddata.get("raw_id")
        if not raw_id:
            continue

        answers: list[str] = []
        for edge in answer_edges_by_dilemma.get(did, []):
            answer_node = graph.get_node(edge.get("to", ""))
            if answer_node:
                ans_id = answer_node.get("raw_id")
                if ans_id:
                    answers.append(ans_id)

        if answers:
            result[raw_id] = sorted(answers)

    return result


def format_retained_entity_ids(
    graph: Graph,
    entity_decisions: list[dict[str, Any]],
) -> str:
    """Format only RETAINED entity IDs for beat generation.

    After entity decisions are serialized, beats should only reference
    entities that were marked as 'retained', not 'cut'. This function
    filters the full entity list to just retained entities.

    Args:
        graph: Graph containing BRAINSTORM entity data.
        entity_decisions: List of EntityDecision dicts from serialization.
            Each must have 'entity_id' and 'disposition' fields.

    Returns:
        Formatted context string with only retained entity IDs, or empty
        string if no retained entities.
    """
    # Build set of cut entity IDs for O(1) lookup
    cut_ids: set[str] = set()
    for decision in entity_decisions:
        entity_id = decision.get("entity_id", "")
        disposition = decision.get("disposition", "retained")
        # Handle both scoped and unscoped IDs
        _, raw_id = parse_scoped_id(entity_id)
        if disposition == "cut":
            cut_ids.add(raw_id)

    # Get all entities from graph, filter out cut ones
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        cat = node.get("entity_type", "unknown")
        raw_id = node.get("raw_id", "")
        if raw_id and raw_id not in cut_ids:
            by_category.setdefault(cat, []).append(raw_id)

    if not by_category:
        return ""

    # Count retained entities
    retained_count = sum(len(ids) for ids in by_category.values())

    lines = [
        "## RETAINED ENTITY IDs (use ONLY these in beats)",
        "",
        f"The following {retained_count} entities are RETAINED for the story.",
        "Do NOT use cut entities - validation will fail.",
        "",
    ]

    for category in ENTITY_CATEGORIES:
        if category in by_category:
            cat_count = len(by_category[category])
            lines.append(f"**{category.title()}s ({cat_count}):**")
            for raw_id in sorted(by_category[category]):
                lines.append(f"  - `{category}::{raw_id}`")
            lines.append("")

    return "\n".join(lines)


def format_summarize_manifest(graph: Graph) -> dict[str, str]:
    """Format entity and dilemma manifests for SEED summarize prompt.

    Returns bullet lists of entity/dilemma IDs that the summarize phase must
    make decisions about (retain/cut for entities, explored/unexplored for
    dilemmas). Simpler than serialize manifest - just lists IDs without
    validation context.

    Note: Entities with unknown category (not in ENTITY_CATEGORIES) are
    excluded from the manifest since they shouldn't appear in BRAINSTORM output.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Dict with 'entity_manifest' and 'dilemma_manifest' strings.
    """
    # Collect entity IDs grouped by category
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        # Support both 'category' (new) and 'entity_type' (legacy) field names
        cat = node.get("category") or node.get("entity_type", "unknown")
        raw_id = node.get("raw_id", "")
        if raw_id:
            by_category.setdefault(cat, []).append(raw_id)

    # Format entity manifest (only standard categories)
    entity_lines: list[str] = []
    for category in ENTITY_CATEGORIES:
        if category in by_category:
            entity_lines.append(f"**{category.title()}s:**")
            for raw_id in sorted(by_category[category]):
                entity_lines.append(f"  - `{category}::{raw_id}`")
            entity_lines.append("")  # Blank line between categories

    # Collect dilemma IDs
    dilemmas = graph.get_nodes_by_type("dilemma")
    dilemma_lines: list[str] = []
    for _did, ddata in sorted(dilemmas.items()):
        raw_id = ddata.get("raw_id")
        if raw_id:
            dilemma_lines.append(f"- `{normalize_scoped_id(raw_id, SCOPE_DILEMMA)}`")

    return {
        "entity_manifest": "\n".join(entity_lines) if entity_lines else "(No entities)",
        "dilemma_manifest": "\n".join(dilemma_lines) if dilemma_lines else "(No dilemmas)",
    }


def format_dilemma_analysis_context(
    seed_output: SeedOutput,
    graph: Graph | None = None,
) -> str:
    """Format surviving dilemmas with narrative context for convergence analysis.

    Builds a rich brief per dilemma including question, stakes, path
    descriptions, and consequence effects. This gives the LLM the signal
    needed to distinguish hard/soft/flavor policies. Used as the ``brief``
    parameter when calling ``serialize_to_artifact`` for Section 7.

    Args:
        seed_output: Pruned SEED output with surviving dilemmas and paths.
        graph: Graph containing brainstorm dilemma nodes (for question,
            why_it_matters). If ``None``, falls back to bare listings.

    Returns:
        Formatted markdown context, or empty string if no dilemmas.
    """
    if not seed_output.dilemmas:
        return ""

    paths_per_dilemma = count_paths_per_dilemma(seed_output)

    # Build path lookup: dilemma raw_id → list of paths
    paths_by_dilemma: dict[str, list[Path]] = {}
    for p in seed_output.paths:
        d_raw = strip_scope_prefix(p.dilemma_id)
        paths_by_dilemma.setdefault(d_raw, []).append(p)

    # Build consequence lookup: normalized raw_id → Consequence
    cons_by_id: dict[str, Consequence] = {}
    for c in seed_output.consequences:
        cons_by_id[strip_scope_prefix(c.consequence_id)] = c

    dilemma_blocks: list[str] = []
    for d in sorted(seed_output.dilemmas, key=lambda x: x.dilemma_id):
        raw_id = strip_scope_prefix(d.dilemma_id)
        path_count = paths_per_dilemma.get(raw_id, 0)

        block_lines: list[str] = [f"### `dilemma::{raw_id}`"]

        # Fetch brainstorm data from graph if available
        if graph is not None:
            node_id = f"{SCOPE_DILEMMA}::{raw_id}"
            node = graph.get_node(node_id)
            if node is not None:
                question = node.get("question", "")
                if question:
                    block_lines.append(f"**Question:** {question}")
                why = node.get("why_it_matters", "")
                if why:
                    block_lines.append(f"**Stakes:** {why}")

        # Path details with consequences
        dilemma_paths = paths_by_dilemma.get(raw_id, [])
        block_lines.append(f"**Paths ({path_count}):**")
        for p in sorted(dilemma_paths, key=lambda x: x.answer_id):
            block_lines.append(f"  - `{p.answer_id}` [{p.path_importance}]: {p.description}")
            # Collect consequence effects for this path
            effects: list[str] = []
            for cid in p.consequence_ids:
                cons = cons_by_id.get(strip_scope_prefix(cid))
                if cons is not None:
                    for eff in cons.narrative_effects:
                        if eff:
                            effects.append(eff)
            if effects:
                block_lines.append(f"    Effects: {' | '.join(effects)}")

        # Show unexplored answers so the LLM can reason about divergence
        if d.unexplored:
            block_lines.append(f"**Unexplored answers:** {', '.join(d.unexplored)}")

        if not dilemma_paths:
            explored = ", ".join(d.explored) if d.explored else "(none)"
            block_lines.append(f"  (no paths yet — explored: [{explored}])")

        dilemma_blocks.append("\n".join(block_lines))

    valid_ids = [f"`dilemma::{strip_scope_prefix(d.dilemma_id)}`" for d in seed_output.dilemmas]

    # Fetch story tone from DREAM vision node
    tone_lines: list[str] = []
    if graph is not None:
        vision = graph.get_node("vision")
        if vision is not None:
            parts: list[str] = []
            if vision.get("genre"):
                parts.append(f"**Genre:** {vision['genre']}")
            tone_val = vision.get("tone")
            if isinstance(tone_val, list) and tone_val:
                parts.append(f"**Tone:** {', '.join(tone_val)}")
            if parts:
                tone_lines = ["## Story Tone", "", *parts, ""]

    lines = [
        "## Dilemma Convergence Brief",
        "",
        "Classify each dilemma below. Target 1-3 `hard` dilemmas per story.",
        "Start by finding the strongest dilemma and classifying it `hard`.",
        "Use `soft` for most others. Use `flavor` only for cosmetic differences.",
        "",
    ]
    if tone_lines:
        lines.extend(tone_lines)
    lines.extend(
        [
            *dilemma_blocks,
            "",
            "### Valid Dilemma IDs",
            "",
            "You MUST use only these dilemma IDs: " + ", ".join(sorted(valid_ids)),
            "",
        ]
    )
    return "\n".join(lines)


def format_interaction_candidates_context(
    seed_output: SeedOutput,
    graph: Graph,
) -> str:
    """Format pre-filtered candidate pairs for interaction analysis (Section 8).

    Reads ``anchored_to`` edges from brainstorm dilemma nodes in the graph,
    filters to surviving dilemmas, and computes candidate pairs that share
    at least one central entity.

    Args:
        seed_output: Pruned SEED output with surviving dilemmas.
        graph: Graph containing brainstorm dilemma nodes with anchored_to edges.

    Returns:
        Formatted markdown context with candidate pairs, or a message
        indicating no candidates if fewer than 2 dilemmas or no shared entities.
    """
    surviving_ids = {strip_scope_prefix(d.dilemma_id) for d in seed_output.dilemmas}

    if len(surviving_ids) < 2:
        return "No candidate pairs — fewer than 2 surviving dilemmas. Return an empty list."

    # Read anchored_to edges from graph dilemma nodes
    dilemma_entities: dict[str, set[str]] = {}
    for raw_id in sorted(surviving_ids):
        node_id = f"{SCOPE_DILEMMA}::{raw_id}"
        if graph.get_node(node_id) is None:
            continue
        edges = graph.get_edges(from_id=node_id, edge_type="anchored_to")
        # Strip scope prefixes for readability
        dilemma_entities[raw_id] = {strip_scope_prefix(e["to"]) for e in edges}

    # Compute candidate pairs (shared central entities)
    candidate_pairs: list[tuple[str, str, list[str]]] = []
    sorted_ids = sorted(dilemma_entities.keys())
    for i, id_a in enumerate(sorted_ids):
        for id_b in sorted_ids[i + 1 :]:
            shared = dilemma_entities[id_a] & dilemma_entities[id_b]
            if shared:
                candidate_pairs.append((id_a, id_b, sorted(shared)))

    if not candidate_pairs:
        return "No candidate pairs — no dilemmas share central entities. Return an empty list."

    pair_lines = [
        f"- `dilemma::{a}` + `dilemma::{b}` (shared: {', '.join(entities)})"
        for a, b, entities in candidate_pairs
    ]

    valid_ids = [f"`dilemma::{rid}`" for rid in sorted(surviving_ids)]

    lines = [
        "## Interaction Candidates",
        "",
        "Only consider these pre-filtered pairs (they share central entities).",
        "Do NOT invent pairs outside this list.",
        "",
        "### Candidate Pairs",
        "",
        *pair_lines,
        "",
        "### Valid Dilemma IDs",
        "",
        "You MUST use only these dilemma IDs: " + ", ".join(valid_ids),
        "",
    ]
    return "\n".join(lines)


def check_structural_completeness(
    output: dict[str, Any],
    expected: dict[str, int],
) -> list[tuple[str, str]]:
    """Check SEED output structural completeness using count-based validation.

    This is a fast pre-check before expensive semantic validation. It catches
    obvious completeness issues (wrong count of decisions) without parsing IDs.

    Args:
        output: SEED output dict with 'entities' and 'dilemmas' arrays.
        expected: Dict from get_expected_counts() with expected counts.
            Values must be non-negative integers.

    Returns:
        List of (field_path, issue) tuples for any completeness errors.
        Empty list if counts match.

    Raises:
        ValueError: If expected counts contain negative values.
    """
    errors: list[tuple[str, str]] = []

    # Validate expected counts are non-negative (defensive check)
    for field, count in expected.items():
        if count < 0:
            raise ValueError(f"Expected count for '{field}' cannot be negative: {count}")

    # Check counts for each tracked field
    for field in ("entities", "dilemmas"):
        actual = len(output.get(field, []))
        expected_count = expected.get(field, 0)
        if actual != expected_count:
            errors.append(
                (field, f"Expected {expected_count} {field[:-1]} decisions, got {actual}")
            )

    return errors
