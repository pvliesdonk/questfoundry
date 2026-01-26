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
SCOPE_ENTITY = "entity"
SCOPE_TENSION = "tension"
SCOPE_THREAD = "thread"

# Development state names (computed, not stored)
STATE_COMMITTED = "committed"
STATE_DEFERRED = "deferred"
STATE_LATENT = "latent"


def count_threads_per_tension(seed_output: SeedOutput) -> dict[str, int]:
    """Count how many threads exist for each tension.

    This is the authoritative source for thread existence - used to derive
    development state (committed vs deferred) and compute arc counts.

    Args:
        seed_output: The SEED output containing threads.

    Returns:
        Dict mapping tension_id (raw, unscoped) to count of threads.
    """
    counts: dict[str, int] = {}
    for thread in seed_output.threads:
        tid = thread.tension_id
        # Handle scoped IDs (tension::foo -> foo)
        if "::" in tid:
            tid = tid.split("::", 1)[1]
        counts[tid] = counts.get(tid, 0) + 1
    return counts


def get_tension_development_states(
    seed_output: SeedOutput,
) -> dict[str, dict[str, str]]:
    """Compute development states for all alternatives in a SEED output.

    Development states are derived from comparing the `considered` field
    (LLM intent) against actual thread existence:
    - **committed**: Alternative in `considered` AND has a thread
    - **deferred**: Alternative in `considered` but NO thread (pruned)
    - **latent**: Alternative not in `considered` (never intended for exploration)

    Args:
        seed_output: The SEED output containing tension decisions and threads.

    Returns:
        Dict mapping tension_id to dict mapping alternative_id to state.
        Example: {"mentor_trust": {"protector": "committed", "manipulator": "deferred"}}
    """
    # Build lookup of which alternatives have threads
    # thread.alternative_id is the raw local ID (not prefixed)
    alt_has_thread: dict[str, set[str]] = {}
    for thread in seed_output.threads:
        tid = thread.tension_id
        # Handle scoped IDs
        if "::" in tid:
            tid = tid.split("::", 1)[1]
        if tid not in alt_has_thread:
            alt_has_thread[tid] = set()
        alt_has_thread[tid].add(thread.alternative_id)

    # Compute states for each tension's alternatives
    states: dict[str, dict[str, str]] = {}
    for tension in seed_output.tensions:
        tid = tension.tension_id
        # Handle scoped IDs
        if "::" in tid:
            tid = tid.split("::", 1)[1]

        tension_states: dict[str, str] = {}
        considered_set = set(tension.considered)
        implicit_set = set(tension.implicit)
        thread_alts = alt_has_thread.get(tid, set())

        # Process all known alternatives (from considered + implicit)
        all_alts = considered_set | implicit_set
        for alt_id in all_alts:
            if alt_id in considered_set:
                if alt_id in thread_alts:
                    tension_states[alt_id] = STATE_COMMITTED
                else:
                    tension_states[alt_id] = STATE_DEFERRED
            else:
                # In implicit, never considered
                tension_states[alt_id] = STATE_LATENT

        states[tid] = tension_states

    return states


def parse_scoped_id(scoped_id: str) -> tuple[str, str]:
    """Parse 'type::raw_id' into (type, raw_id).

    Scoped IDs use '::' as delimiter to disambiguate ID types for the LLM.
    This prevents confusion between entity IDs, tension IDs, and thread IDs.

    Args:
        scoped_id: An ID string, optionally scoped (e.g., 'entity::hero' or 'hero').

    Returns:
        Tuple of (scope_type, raw_id). Returns ('', raw_id) if unscoped.

    Examples:
        >>> parse_scoped_id("entity::hero")
        ('entity', 'hero')
        >>> parse_scoped_id("thread::host_motive")
        ('thread', 'host_motive')
        >>> parse_scoped_id("hero")
        ('', 'hero')
    """
    if "::" in scoped_id:
        parts = scoped_id.split("::", 1)
        return parts[0], parts[1]
    return "", scoped_id


def format_scoped_id(scope: str, raw_id: str) -> str:
    """Format a raw ID with its scope prefix.

    Args:
        scope: The scope type (e.g., 'entity', 'tension', 'thread').
        raw_id: The raw ID without scope.

    Returns:
        Scoped ID string (e.g., 'entity::hero').
    """
    return f"{scope}::{raw_id}"


def normalize_scoped_id(raw_id: str, scope: str) -> str:
    """Ensure an ID has the scope prefix, adding it if missing.

    Args:
        raw_id: ID that may or may not already have the scope prefix.
        scope: The scope type (e.g., 'tension', 'thread').

    Returns:
        ID with scope prefix guaranteed (e.g., 'tension::mentor_trust').
    """
    prefix = f"{scope}::"
    return raw_id if raw_id.startswith(prefix) else f"{prefix}{raw_id}"


def format_thread_ids_context(threads: list[dict[str, Any]]) -> str:
    """Format thread IDs for beat serialization with inline constraints.

    Includes thread→tension mapping so the model knows which tension_id
    to use in tension_impacts for each beat's thread.

    Small models don't "look back" at referenced sections, so we
    embed constraints at the point of use.

    Args:
        threads: List of thread dicts from serialized ThreadsSection.

    Returns:
        Formatted context string with thread IDs, or empty string if none.
    """
    if not threads:
        return ""

    # Sort for deterministic output across runs
    thread_ids = sorted(tid for t in threads if (tid := t.get("thread_id")))

    if not thread_ids:
        return ""

    # Pipe-delimited for easy scanning, with thread:: scope prefix
    id_list = " | ".join(f"`{format_scoped_id(SCOPE_THREAD, tid)}`" for tid in thread_ids)

    lines = [
        "## VALID THREAD IDs (copy exactly, no modifications)",
        "",
        f"Allowed: {id_list}",
        "",
    ]

    # Build thread→tension mapping for tension_impacts guidance
    thread_tension_pairs = []
    for t in sorted(threads, key=lambda x: x.get("thread_id", "")):
        tid = t.get("thread_id")
        tension_id = t.get("tension_id", "")
        if tid and tension_id:
            # Strip scope prefix if present for clean display
            raw_tension = tension_id.split("::", 1)[-1] if "::" in tension_id else tension_id
            thread_tension_pairs.append((tid, raw_tension))

    if thread_tension_pairs:
        lines.append("## THREAD → TENSION MAPPING (use in tension_impacts)")
        lines.append("Each beat's tension_impacts should use its thread's parent tension:")
        lines.append("")
        for tid, tension in thread_tension_pairs:
            lines.append(
                f"  - `{format_scoped_id(SCOPE_THREAD, tid)}` → "
                f"`{format_scoped_id(SCOPE_TENSION, tension)}`"
            )
        lines.append("")

    lines.extend(
        [
            "Rules:",
            "- Use ONLY IDs from the list above in the `threads` array",
            "- Include the `thread::` prefix in your output",
            "- For tension_impacts, use the tension from the mapping above",
            "- Do NOT derive IDs from tension concepts",
            "- Do NOT use entity IDs as tension_ids",
            "- If a concept has no matching thread, omit it - do NOT invent an ID",
            "",
            "WRONG (will fail validation):",
            "- `clock_distortion` - NOT a valid thread ID (derived from concept)",
            "- `host_motive` - WRONG: missing scope prefix, use `thread::host_motive`",
            "- `tension::seed_of_stillness` - WRONG: entity ID used as tension",
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
        ("Thread IDs", "valid_thread_ids"),
        ("Tension IDs", "valid_tension_ids"),
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

    Groups entities by category and lists tensions with their alternatives,
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
    tension_count = counts["tensions"]

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
                    lines.append(f"  - `{format_scoped_id(SCOPE_ENTITY, raw_id)}`")
                lines.append("")

    # Tensions with alternatives
    tensions = graph.get_nodes_by_type("tension")
    if tensions:
        # Pre-build alt edges map to avoid O(T*E) lookups
        alt_edges_by_tension: dict[str, list[dict[str, Any]]] = {}
        for edge in graph.get_edges(edge_type="has_alternative"):
            from_id = edge.get("from")
            if from_id:
                alt_edges_by_tension.setdefault(from_id, []).append(edge)

        lines.append(f"### Tension IDs (TOTAL: {tension_count} - generate decision for ALL)")
        lines.append("Format: tension::id → [alternative_ids]")
        lines.append("Note: Every tension_id contains `_or_` — entity IDs never do.")
        lines.append("")

        for tid, tdata in sorted(tensions.items()):
            raw_id = tdata.get("raw_id")
            if not raw_id:
                continue

            alts = []
            for edge in alt_edges_by_tension.get(tid, []):
                alt_node = graph.get_node(edge.get("to", ""))
                if alt_node:
                    alt_id = alt_node.get("raw_id")
                    if alt_id:
                        default = " (default)" if alt_node.get("is_default_path") else ""
                        alts.append(f"`{alt_id}`{default}")

            if alts:
                # Sort alternatives for deterministic output
                alts.sort()
                lines.append(f"- `{format_scoped_id(SCOPE_TENSION, raw_id)}` → [{', '.join(alts)}]")

        lines.append("")

    # Generation requirements with counts (handle singular/plural grammar)
    entity_word = "item" if total_entity_count == 1 else "items"
    tension_word = "item" if tension_count == 1 else "items"

    lines.extend(
        [
            "### Generation Requirements (CRITICAL)",
            f"- Generate EXACTLY {total_entity_count} entity decisions (one per entity above)",
            f"- Generate EXACTLY {tension_count} tension decisions (one per tension above)",
            "- Thread `alternative_id` must be from that tension's alternatives list",
            "- Use scoped IDs with `entity::` or `tension::` prefix in your output",
            "",
            "### Verification",
            "Before submitting, COUNT your outputs:",
            f"- entities array should have {total_entity_count} {entity_word}",
            f"- tensions array should have {tension_count} {tension_word}",
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
    tensions = graph.get_nodes_by_type("tension")

    entity_count = sum(1 for node in entities.values() if node.get("raw_id"))
    tension_count = sum(1 for tdata in tensions.values() if tdata.get("raw_id"))

    return {
        "entities": entity_count,
        "tensions": tension_count,
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
                lines.append(f"  - `{format_scoped_id(SCOPE_ENTITY, raw_id)}`")
            lines.append("")

    return "\n".join(lines)


def format_summarize_manifest(graph: Graph) -> dict[str, str]:
    """Format entity and tension manifests for SEED summarize prompt.

    Returns bullet lists of entity/tension IDs that the summarize phase must
    make decisions about (retain/cut for entities, explored/implicit for
    tensions). Simpler than serialize manifest - just lists IDs without
    validation context.

    Note: Entities with unknown entity_type (not in _ENTITY_CATEGORIES) are
    excluded from the manifest since they shouldn't appear in BRAINSTORM output.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Dict with 'entity_manifest' and 'tension_manifest' strings.
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
                entity_lines.append(f"  - `{format_scoped_id(SCOPE_ENTITY, raw_id)}`")
            entity_lines.append("")  # Blank line between categories

    # Collect tension IDs
    tensions = graph.get_nodes_by_type("tension")
    tension_lines: list[str] = []
    for _tid, tdata in sorted(tensions.items()):
        raw_id = tdata.get("raw_id")
        if raw_id:
            tension_lines.append(f"- `{format_scoped_id(SCOPE_TENSION, raw_id)}`")

    return {
        "entity_manifest": "\n".join(entity_lines) if entity_lines else "(No entities)",
        "tension_manifest": "\n".join(tension_lines) if tension_lines else "(No tensions)",
    }


def check_structural_completeness(
    output: dict[str, Any],
    expected: dict[str, int],
) -> list[tuple[str, str]]:
    """Check SEED output structural completeness using count-based validation.

    This is a fast pre-check before expensive semantic validation. It catches
    obvious completeness issues (wrong count of decisions) without parsing IDs.

    Args:
        output: SEED output dict with 'entities' and 'tensions' arrays.
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
    for field in ("entities", "tensions"):
        actual = len(output.get(field, []))
        expected_count = expected.get(field, 0)
        if actual != expected_count:
            errors.append(
                (field, f"Expected {expected_count} {field[:-1]} decisions, got {actual}")
            )

    return errors
