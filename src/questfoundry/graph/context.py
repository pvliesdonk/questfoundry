"""Graph context formatting for LLM prompts.

Provides functions to format graph data as context for LLM serialization,
giving the model authoritative lists of valid IDs to reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.grow_context import format_grow_valid_ids

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph
    from questfoundry.models.seed import SeedOutput

# Standard entity categories for BRAINSTORM/SEED stages
_ENTITY_CATEGORIES = ["character", "location", "object", "faction"]

# Scope prefixes for typed IDs (disambiguates ID types for LLM)
# Full word prefixes for semantic clarity (recommended by prompt engineering analysis)
SCOPE_ENTITY = "entity"
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

    Development states are derived from comparing the `considered` field
    (LLM intent) against actual path existence:
    - **committed**: Answer in `considered` AND has a path
    - **deferred**: Answer in `considered` but NO path (pruned)
    - **latent**: Answer not in `considered` (never intended for exploration)

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
        considered_set = set(dilemma.considered)
        implicit_set = set(dilemma.implicit)
        path_answers = answer_has_path.get(did, set())

        # Process all known answers (from considered + implicit)
        all_answers = considered_set | implicit_set
        for ans_id in all_answers:
            if ans_id in considered_set:
                if ans_id in path_answers:
                    dilemma_states[ans_id] = STATE_COMMITTED
                else:
                    dilemma_states[ans_id] = STATE_DEFERRED
            else:
                # In implicit, never considered
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
        scope: The scope type (e.g., 'dilemma', 'path', 'entity').

    Returns:
        ID with scope prefix guaranteed (e.g., 'dilemma::mentor_trust').
    """
    prefix = f"{scope}::"
    return raw_id if raw_id.startswith(prefix) else f"{prefix}{raw_id}"


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

    After dilemmas are serialized, each dilemma has ``considered`` and
    ``implicit`` answer lists.  Injecting these before the paths section
    lets the model know exactly which answer_id values are valid for each
    dilemma, preventing phantom answer references.

    Args:
        dilemmas: List of dilemma decision dicts from serialized output.

    Returns:
        Formatted manifest string, or empty string if no dilemmas.
    """
    if not dilemmas:
        return ""

    lines = [
        "## Valid Answer IDs per Dilemma",
        "",
        "Each path's `answer_id` MUST be one of the `considered` IDs below.",
        "Do NOT invent answer IDs or use `implicit` IDs as path answer_ids.",
        "",
    ]

    for d in sorted(dilemmas, key=lambda x: x.get("dilemma_id", "")):
        dilemma_id = d.get("dilemma_id", "")
        if not dilemma_id:
            continue
        scoped = normalize_scoped_id(strip_scope_prefix(dilemma_id), SCOPE_DILEMMA)
        considered = d.get("considered", [])
        implicit = d.get("implicit", [])
        lines.append(f"- `{scoped}` -> considered: {considered}, implicit: {implicit}")

    lines.append("")
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


def format_valid_ids_context(graph: Graph, stage: str) -> str:
    """Format valid IDs as context for LLM serialization.

    Provides the authoritative list of IDs the LLM must use.
    This prevents phantom ID references by showing valid options upfront.

    Args:
        graph: Graph containing nodes from previous stages.
        stage: Current stage name ("seed", "grow", etc.).

    Returns:
        Formatted context string, or empty string if not applicable.
    """
    if stage == "seed":
        return _format_seed_valid_ids(graph)
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


def _format_seed_valid_ids(graph: Graph) -> str:
    """Format BRAINSTORM IDs for SEED serialization.

    Groups entities by category and lists dilemmas with their answers,
    making it clear which IDs are valid for the SEED stage to reference.
    Includes counts to enable completeness validation.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Formatted context string with valid IDs and counts.
    """
    # Get expected counts from canonical source (DRY principle)
    counts = get_expected_counts(graph)
    total_entity_count = counts["entities"]
    dilemma_count = counts["dilemmas"]

    lines = [
        "## VALID IDS MANIFEST - GENERATE FOR ALL",
        "",
        "You MUST generate a decision for EVERY ID listed below.",
        "Missing items WILL cause validation failure.",
        "",
    ]

    # Group entities by type (character, location, object, faction)
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        cat = node.get("entity_type", "unknown")
        raw_id = node.get("raw_id", "")
        if raw_id:  # Only include entities with valid raw_id
            by_category.setdefault(cat, []).append(raw_id)

    if by_category:
        lines.append(f"### Entity IDs (TOTAL: {total_entity_count} - generate decision for ALL)")
        lines.append("Use these for `entity_id`, `entities`, and `location` fields:")
        lines.append("")

        for category in _ENTITY_CATEGORIES:
            if category in by_category:
                cat_count = len(by_category[category])
                lines.append(f"**{category.title()}s ({cat_count}):**")
                for raw_id in sorted(by_category[category]):
                    lines.append(f"  - `{normalize_scoped_id(raw_id, SCOPE_ENTITY)}`")
                lines.append("")

    # Dilemmas with answers
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
                        default = " (default)" if answer_node.get("is_default_path") else ""
                        answers.append(f"`{ans_id}`{default}")

            if answers:
                # Sort answers for deterministic output
                answers.sort()
                lines.append(
                    f"- `{normalize_scoped_id(raw_id, SCOPE_DILEMMA)}` → [{', '.join(answers)}]"
                )

        lines.append("")

    # Generation requirements with counts (handle singular/plural grammar)
    entity_word = "item" if total_entity_count == 1 else "items"
    dilemma_word = "item" if dilemma_count == 1 else "items"

    lines.extend(
        [
            "### Generation Requirements (CRITICAL)",
            f"- Generate EXACTLY {total_entity_count} entity decisions (one per entity above)",
            f"- Generate EXACTLY {dilemma_count} dilemma decisions (one per dilemma above)",
            "- Path `answer_id` must be from that dilemma's answers list",
            "- Use scoped IDs with `entity::` or `dilemma::` prefix in your output",
            "",
            "### Verification",
            "Before submitting, COUNT your outputs:",
            f"- entities array should have {total_entity_count} {entity_word}",
            f"- dilemmas array should have {dilemma_count} {dilemma_word}",
        ]
    )

    return "\n".join(lines)


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

    for category in _ENTITY_CATEGORIES:
        if category in by_category:
            cat_count = len(by_category[category])
            lines.append(f"**{category.title()}s ({cat_count}):**")
            for raw_id in sorted(by_category[category]):
                lines.append(f"  - `{normalize_scoped_id(raw_id, SCOPE_ENTITY)}`")
            lines.append("")

    return "\n".join(lines)


def format_summarize_manifest(graph: Graph) -> dict[str, str]:
    """Format entity and dilemma manifests for SEED summarize prompt.

    Returns bullet lists of entity/dilemma IDs that the summarize phase must
    make decisions about (retain/cut for entities, explored/implicit for
    dilemmas). Simpler than serialize manifest - just lists IDs without
    validation context.

    Note: Entities with unknown entity_type (not in _ENTITY_CATEGORIES) are
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
        cat = node.get("entity_type", "unknown")
        raw_id = node.get("raw_id", "")
        if raw_id:
            by_category.setdefault(cat, []).append(raw_id)

    # Format entity manifest (only standard categories)
    entity_lines: list[str] = []
    for category in _ENTITY_CATEGORIES:
        if category in by_category:
            entity_lines.append(f"**{category.title()}s:**")
            for raw_id in sorted(by_category[category]):
                entity_lines.append(f"  - `{normalize_scoped_id(raw_id, SCOPE_ENTITY)}`")
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
