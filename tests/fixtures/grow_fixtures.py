"""Factory functions for GROW stage test graphs.

These build pre-populated Graph objects with the node/edge structures
that GROW expects from earlier stages (BRAINSTORM, SEED).

Graph structure conventions:
- Entity nodes: entity::{raw_id}
- Dilemma nodes: dilemma::{raw_id}
- Path nodes: path::{raw_id}
- Beat nodes: beat::{raw_id}
- Consequence nodes: consequence::{raw_id}
- Edges: belongs_to (beat→path), explores (path→dilemma),
         has_consequence (path→consequence), requires (beat→beat)
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph

# ---------------------------------------------------------------------------
# DREAM + BRAINSTORM + SEED upstream baseline helpers
#
# These populate the mandatory upstream contract that validate_grow_output
# checks via _check_upstream_contract → validate_seed_output.
#
# Usage: call seed_upstream_baseline(graph) on any graph that will be run
# through the full GROW stage (not needed for unit tests that only exercise
# individual algorithms).
# ---------------------------------------------------------------------------


def seed_dream_baseline(graph: Graph) -> None:
    """Add a minimal DREAM vision node (R-1.7)."""
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["forbidden knowledge"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )


def seed_brainstorm_baseline(graph: Graph) -> None:
    """Add minimal BRAINSTORM entities + dilemma satisfying R-1.1/R-2.1/R-2.4.

    Creates:
      - 2 character entities + 2 location entities (satisfies R-2.4: ≥2 locations)
      - 1 dilemma with 2 answers (1 canonical)
    """
    for eid, cat, name in [
        ("character::kay", "character", "Kay"),
        ("character::mentor", "character", "Mentor"),
        ("location::archive", "location", "Archive"),
        ("location::depths", "location", "Forbidden Depths"),
    ]:
        graph.create_node(
            eid,
            {
                "type": "entity",
                "raw_id": eid.split("::", 1)[-1],
                "name": name,
                "category": cat,
                "concept": "x",
                "disposition": "retained",
            },
        )
    graph.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Trust?",
            "why_it_matters": "stakes",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    for ans, is_canon in [("protector", True), ("manipulator", False)]:
        ans_id = f"dilemma::mentor_trust::alt::{ans}"
        graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": ans,
                "description": f"d-{ans}",
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        graph.add_edge("has_answer", "dilemma::mentor_trust", ans_id)
    graph.add_edge("anchored_to", "dilemma::mentor_trust", "character::mentor")


def seed_seed_baseline(graph: Graph) -> None:
    """Add minimal SEED paths, beats, consequences and Path Freeze node.

    Creates a Y-shaped dilemma (mentor_trust) with:
      - 1 shared pre-commit beat
      - 2 commit beats (one per path)
      - 2 post-commit beats per path
      - seed_freeze approval node
    """
    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        graph.create_node(
            path_id,
            {
                "type": "path",
                "raw_id": f"mentor_trust__{ans}",
                "dilemma_id": "dilemma::mentor_trust",
                "is_canonical": ans == "protector",
            },
        )
        graph.add_edge("explores", path_id, f"dilemma::mentor_trust::alt::{ans}")
        conseq_id = f"consequence::mentor_trust__{ans}"
        graph.create_node(
            conseq_id,
            {
                "type": "consequence",
                "raw_id": f"mentor_trust__{ans}",
                "description": "mentor becomes hostile",
                "ripples": ["faction mistrust rises"],
            },
        )
        graph.add_edge("has_consequence", path_id, conseq_id)

    graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "Mentor delivers warning",
            "entities": ["character::mentor", "character::kay"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}],
        },
    )
    graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__protector")
    graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__manipulator")

    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        commit_id = f"beat::commit_{ans}"
        graph.create_node(
            commit_id,
            {
                "type": "beat",
                "raw_id": f"commit_{ans}",
                "summary": f"Mentor reveals {ans} motive",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", commit_id, path_id)
        for i in range(1, 3):  # 2 post-commit beats
            post_id = f"beat::post_{ans}_{i:02d}"
            graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": f"Post-commit {i} on {ans}",
                    "entities": ["character::mentor"],
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", post_id, path_id)

    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": True})


def seed_upstream_baseline(graph: Graph) -> None:
    """Populate DREAM + BRAINSTORM + SEED upstream baseline on *graph*.

    Call this on any graph that will be run through the full GROW stage so
    that validate_grow_output (which delegates to validate_seed_output) does
    not raise upstream-contract errors.

    Note: This creates a *new* dilemma / entity set (mentor_trust / kay /
    mentor / archive / depths). If your test graph already has its own
    dilemmas and entities, call the three helpers individually and supply
    only the missing pieces, rather than calling this combined helper.
    """
    seed_dream_baseline(graph)
    seed_brainstorm_baseline(graph)
    seed_seed_baseline(graph)


def make_single_dilemma_graph() -> Graph:
    """Create a minimal graph with 1 dilemma, 2 paths, 4 beats.

    Structure:
        dilemma: mentor_trust
        paths: mentor_trust_canonical (canonical), mentor_trust_alt (answer)
        beats: opening, mentor_meet, mentor_commits_canonical, mentor_commits_alt

    Beat ordering (requires edges):
        opening → mentor_meet → mentor_commits_canonical
        opening → mentor_meet → mentor_commits_alt

    Consequences:
        mentor_trusted (path: mentor_trust_canonical)
        mentor_distrusted (path: mentor_trust_alt)

    Returns:
        Populated Graph instance.
    """
    graph = Graph.empty()

    # Entities
    graph.create_node(
        "character::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_category": "character",
            "concept": "A wise mentor",
        },
    )
    graph.create_node(
        "character::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_category": "character",
            "concept": "The protagonist",
        },
    )

    # Dilemma
    graph.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
            "ending_salience": "high",
        },
    )

    # Answers
    graph.create_node(
        "dilemma::mentor_trust::alt::trust_yes",
        {"type": "answer", "raw_id": "trust_yes", "dilemma_id": "mentor_trust"},
    )
    graph.create_node(
        "dilemma::mentor_trust::alt::trust_no",
        {"type": "answer", "raw_id": "trust_no", "dilemma_id": "mentor_trust"},
    )
    graph.add_edge("has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::trust_yes")
    graph.add_edge("has_answer", "dilemma::mentor_trust", "dilemma::mentor_trust::alt::trust_no")

    # Paths
    graph.create_node(
        "path::mentor_trust_canonical",
        {
            "type": "path",
            "raw_id": "mentor_trust_canonical",
            "dilemma_id": "dilemma::mentor_trust",
            "answer_id": "trust_yes",
            "path_importance": "major",
            "is_canonical": True,
        },
    )
    graph.create_node(
        "path::mentor_trust_alt",
        {
            "type": "path",
            "raw_id": "mentor_trust_alt",
            "dilemma_id": "dilemma::mentor_trust",
            "answer_id": "trust_no",
            "path_importance": "major",
            "is_canonical": False,
        },
    )

    # Path → Answer edges (explores)
    graph.add_edge(
        "explores", "path::mentor_trust_canonical", "dilemma::mentor_trust::alt::trust_yes"
    )
    graph.add_edge("explores", "path::mentor_trust_alt", "dilemma::mentor_trust::alt::trust_no")

    # Beats
    graph.create_node(
        "beat::opening",
        {
            "type": "beat",
            "raw_id": "opening",
            "summary": "The story begins.",
            "paths": ["mentor_trust_canonical", "mentor_trust_alt"],
        },
    )
    graph.create_node(
        "beat::mentor_meet",
        {
            "type": "beat",
            "raw_id": "mentor_meet",
            "summary": "Hero meets the mentor.",
            "paths": ["mentor_trust_canonical", "mentor_trust_alt"],
        },
    )
    graph.create_node(
        "beat::mentor_commits_canonical",
        {
            "type": "beat",
            "raw_id": "mentor_commits_canonical",
            "summary": "Hero trusts the mentor fully.",
            "paths": ["mentor_trust_canonical"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        },
    )
    graph.create_node(
        "beat::mentor_commits_alt",
        {
            "type": "beat",
            "raw_id": "mentor_commits_alt",
            "summary": "Hero rejects the mentor.",
            "paths": ["mentor_trust_alt"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        },
    )

    # Beat → Path edges (belongs_to)
    graph.add_edge("belongs_to", "beat::opening", "path::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::opening", "path::mentor_trust_alt")
    graph.add_edge("belongs_to", "beat::mentor_meet", "path::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::mentor_meet", "path::mentor_trust_alt")
    graph.add_edge("belongs_to", "beat::mentor_commits_canonical", "path::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::mentor_commits_alt", "path::mentor_trust_alt")

    # Beat ordering (requires edges): opening → mentor_meet → commits
    graph.add_edge("predecessor", "beat::mentor_meet", "beat::opening")
    graph.add_edge("predecessor", "beat::mentor_commits_canonical", "beat::mentor_meet")
    graph.add_edge("predecessor", "beat::mentor_commits_alt", "beat::mentor_meet")

    # Consequences
    graph.create_node(
        "consequence::mentor_trusted",
        {
            "type": "consequence",
            "raw_id": "mentor_trusted",
            "path_id": "mentor_trust_canonical",
            "description": "The mentor becomes an ally.",
        },
    )
    graph.create_node(
        "consequence::mentor_distrusted",
        {
            "type": "consequence",
            "raw_id": "mentor_distrusted",
            "path_id": "mentor_trust_alt",
            "description": "The mentor becomes an adversary.",
        },
    )
    graph.add_edge("has_consequence", "path::mentor_trust_canonical", "consequence::mentor_trusted")
    graph.add_edge("has_consequence", "path::mentor_trust_alt", "consequence::mentor_distrusted")

    graph.set_last_stage("seed")
    return graph


def make_two_dilemma_graph() -> Graph:
    """Create a graph with 2 dilemmas, 4 paths, 8 beats.

    Structure:
        dilemma: mentor_trust (2 paths: canonical, alt)
        dilemma: artifact_quest (2 paths: canonical, alt)

    Paths:
        mentor_trust_canonical, mentor_trust_alt
        artifact_quest_canonical, artifact_quest_alt

    Beats per path (with shared opening/closing beats):
        opening (all paths)
        mentor_meet (mentor paths)
        artifact_discover (artifact paths)
        mentor_commits_canonical (mentor_trust_canonical)
        mentor_commits_alt (mentor_trust_alt)
        artifact_commits_canonical (artifact_quest_canonical)
        artifact_commits_alt (artifact_quest_alt)
        finale (all paths)

    Beat ordering:
        opening → mentor_meet → mentor_commits_*
        opening → artifact_discover → artifact_commits_*
        mentor_commits_* → finale
        artifact_commits_* → finale

    Returns:
        Populated Graph instance.
    """
    graph = Graph.empty()

    # Entities
    graph.create_node(
        "character::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_category": "character",
            "concept": "A wise mentor",
        },
    )
    graph.create_node(
        "character::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_category": "character",
            "concept": "The protagonist",
        },
    )
    graph.create_node(
        "object::artifact",
        {
            "type": "entity",
            "raw_id": "artifact",
            "entity_category": "object",
            "concept": "A powerful artifact",
        },
    )

    # Dilemmas
    graph.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
            "ending_salience": "high",
        },
    )
    graph.create_node(
        "dilemma::artifact_quest",
        {
            "type": "dilemma",
            "raw_id": "artifact_quest",
            "question": "Does the hero use the artifact for good?",
            "ending_salience": "high",
        },
    )

    # Answers
    for dilemma_id, alt_id in [
        ("mentor_trust", "trust_yes"),
        ("mentor_trust", "trust_no"),
        ("artifact_quest", "use_good"),
        ("artifact_quest", "use_selfish"),
    ]:
        alt_node_id = f"dilemma::{dilemma_id}::alt::{alt_id}"
        graph.create_node(
            alt_node_id,
            {"type": "answer", "raw_id": alt_id, "dilemma_id": dilemma_id},
        )
        graph.add_edge("has_answer", f"dilemma::{dilemma_id}", alt_node_id)

    # Paths
    for path_id, dilemma_id, alt_id, is_canon in [
        ("mentor_trust_canonical", "mentor_trust", "trust_yes", True),
        ("mentor_trust_alt", "mentor_trust", "trust_no", False),
        ("artifact_quest_canonical", "artifact_quest", "use_good", True),
        ("artifact_quest_alt", "artifact_quest", "use_selfish", False),
    ]:
        graph.create_node(
            f"path::{path_id}",
            {
                "type": "path",
                "raw_id": path_id,
                "dilemma_id": f"dilemma::{dilemma_id}",
                "answer_id": alt_id,
                "path_importance": "major",
                "is_canonical": is_canon,
            },
        )
        graph.add_edge("explores", f"path::{path_id}", f"dilemma::{dilemma_id}::alt::{alt_id}")

    # Beats
    all_paths = [
        "mentor_trust_canonical",
        "mentor_trust_alt",
        "artifact_quest_canonical",
        "artifact_quest_alt",
    ]
    mentor_paths = ["mentor_trust_canonical", "mentor_trust_alt"]
    artifact_paths = ["artifact_quest_canonical", "artifact_quest_alt"]

    beats = [
        ("opening", "The story begins.", all_paths, []),
        ("mentor_meet", "Hero meets the mentor.", mentor_paths, []),
        ("artifact_discover", "Hero discovers the artifact.", artifact_paths, []),
        (
            "mentor_commits_canonical",
            "Hero trusts the mentor.",
            ["mentor_trust_canonical"],
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        ),
        (
            "mentor_commits_alt",
            "Hero distrusts the mentor.",
            ["mentor_trust_alt"],
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        ),
        (
            "artifact_commits_canonical",
            "Hero uses artifact for good.",
            ["artifact_quest_canonical"],
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "commits"}],
        ),
        (
            "artifact_commits_alt",
            "Hero uses artifact selfishly.",
            ["artifact_quest_alt"],
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "commits"}],
        ),
        ("finale", "The conclusion.", all_paths, []),
    ]

    for beat_id, summary, paths, impacts in beats:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "paths": paths,
                "dilemma_impacts": impacts,
            },
        )
        for path_id in paths:
            graph.add_edge("belongs_to", f"beat::{beat_id}", f"path::{path_id}")

    # Beat ordering (requires edges)
    # opening → mentor_meet, artifact_discover
    graph.add_edge("predecessor", "beat::mentor_meet", "beat::opening")
    graph.add_edge("predecessor", "beat::artifact_discover", "beat::opening")
    # mentor_meet → mentor_commits_*
    graph.add_edge("predecessor", "beat::mentor_commits_canonical", "beat::mentor_meet")
    graph.add_edge("predecessor", "beat::mentor_commits_alt", "beat::mentor_meet")
    # artifact_discover → artifact_commits_*
    graph.add_edge("predecessor", "beat::artifact_commits_canonical", "beat::artifact_discover")
    graph.add_edge("predecessor", "beat::artifact_commits_alt", "beat::artifact_discover")
    # commits → finale
    graph.add_edge("predecessor", "beat::finale", "beat::mentor_commits_canonical")
    graph.add_edge("predecessor", "beat::finale", "beat::mentor_commits_alt")
    graph.add_edge("predecessor", "beat::finale", "beat::artifact_commits_canonical")
    graph.add_edge("predecessor", "beat::finale", "beat::artifact_commits_alt")

    # Consequences
    for cons_id, path_id, desc in [
        ("mentor_trusted", "mentor_trust_canonical", "Mentor becomes ally."),
        ("mentor_distrusted", "mentor_trust_alt", "Mentor becomes adversary."),
        ("artifact_saved", "artifact_quest_canonical", "World is saved."),
        ("artifact_corrupted", "artifact_quest_alt", "World is corrupted."),
    ]:
        graph.create_node(
            f"consequence::{cons_id}",
            {
                "type": "consequence",
                "raw_id": cons_id,
                "path_id": path_id,
                "description": desc,
            },
        )
        graph.add_edge("has_consequence", f"path::{path_id}", f"consequence::{cons_id}")

    graph.set_last_stage("seed")
    return graph


def make_e2e_fixture_graph() -> Graph:
    """Create a detailed graph for E2E integration testing.

    Structure: 2 dilemmas x 2 paths = 4 arcs with Y-shape forks.

    Dilemmas:
        mentor_trust: Does the hero trust the mentor? (soft — paths converge)
        artifact_quest: Does the hero use the artifact for good? (soft)

    Paths (4):
        mentor_trust_canonical (trust_yes, canonical)
        mentor_trust_alt (trust_no)
        artifact_quest_canonical (use_good, canonical)
        artifact_quest_alt (use_selfish)

    Beat topology (Y-shape per dilemma, structural opening + climax):
        opening [setup, zero belongs_to]
        ├─ mt_encounter (reveals) → mt_test (advances) ┬─ mt_trust   (commits canonical) → mt_trust_post_01   → mt_trust_post_02   → climax
        │                                              └─ mt_distrust (commits alt)      → mt_distrust_post_01 → mt_distrust_post_02 → climax
        └─ aq_discovery (reveals) → aq_trial (advances) ┬─ aq_wield   (commits canonical) → aq_wield_post_01   → aq_wield_post_02   → climax
                                                        └─ aq_corrupt (commits alt)      → aq_corrupt_post_01 → aq_corrupt_post_02 → climax
        climax [epilogue, zero belongs_to]

    Pre-commit beats carry dual `belongs_to` (one edge per path of their
    dilemma); commit and post-commit beats carry single `belongs_to`
    (their own path). opening and climax are structural (zero `belongs_to`,
    zero `dilemma_impacts`).

    Returns:
        Populated Graph instance ready for GROW processing.
    """
    graph = Graph.empty()

    # Vision (R-1.7)
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["power and its cost"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )

    # Entities (R-2.1: name + category; R-1.1: disposition retained)
    for eid, cat, name, concept in [
        ("hero", "character", "Hero", "A young adventurer seeking purpose"),
        ("mentor", "character", "Mentor", "A mysterious sage with hidden agenda"),
        ("artifact", "object", "Crystal", "An ancient crystal of immense power"),
        ("temple", "location", "Temple", "Crumbling temple in the mountains"),
        ("village", "location", "Village", "The hero's home village"),
    ]:
        graph.create_node(
            f"{cat}::{eid}",
            {
                "type": "entity",
                "raw_id": eid,
                "name": name,
                "category": cat,
                "concept": concept,
                "disposition": "retained",
            },
        )

    # Dilemmas (R-3.1 why_it_matters, R-7.1 dilemma_role, R-7.2 residue_weight)
    graph.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
            "why_it_matters": "the hero's judgement shapes every later alliance",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "high",
        },
    )
    graph.create_node(
        "dilemma::artifact_quest",
        {
            "type": "dilemma",
            "raw_id": "artifact_quest",
            "question": "Does the hero use the artifact for good or selfish ends?",
            "why_it_matters": "the artifact's use defines the world's fate",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "high",
        },
    )

    # Dilemma anchors (R-3.6)
    graph.add_edge("anchored_to", "dilemma::mentor_trust", "character::mentor")
    graph.add_edge("anchored_to", "dilemma::artifact_quest", "object::artifact")

    # Answers (R-3.5 description, R-3.4 exactly one canonical)
    answer_defs = [
        ("mentor_trust", "trust_yes", "Trust the mentor completely", True),
        ("mentor_trust", "trust_no", "Reject the mentor and go alone", False),
        ("artifact_quest", "use_good", "Use the artifact for good", True),
        ("artifact_quest", "use_selfish", "Use the artifact for personal gain", False),
    ]
    for dilemma_id, alt_id, desc, is_canon in answer_defs:
        alt_node_id = f"dilemma::{dilemma_id}::alt::{alt_id}"
        graph.create_node(
            alt_node_id,
            {
                "type": "answer",
                "raw_id": alt_id,
                "dilemma_id": dilemma_id,
                "description": desc,
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        graph.add_edge("has_answer", f"dilemma::{dilemma_id}", alt_node_id)

    # Paths
    path_defs = [
        ("mentor_trust_canonical", "mentor_trust", "trust_yes", True),
        ("mentor_trust_alt", "mentor_trust", "trust_no", False),
        ("artifact_quest_canonical", "artifact_quest", "use_good", True),
        ("artifact_quest_alt", "artifact_quest", "use_selfish", False),
    ]
    mentor_path_ids = [t[0] for t in path_defs if "mentor" in t[0]]
    artifact_path_ids = [t[0] for t in path_defs if "artifact" in t[0]]

    for path_id, dilemma_id, alt_id, is_canon in path_defs:
        graph.create_node(
            f"path::{path_id}",
            {
                "type": "path",
                "raw_id": path_id,
                "dilemma_id": f"dilemma::{dilemma_id}",
                "answer_id": alt_id,
                "path_importance": "major",
                "is_canonical": is_canon,
            },
        )
        graph.add_edge("explores", f"path::{path_id}", f"dilemma::{dilemma_id}::alt::{alt_id}")

    # Beats (R-3.13 entities list, R-3.6/R-3.9 same-dilemma belongs_to,
    # R-3.12 2-4 post-commit beats per path).  Structural beats use
    # role=setup/epilogue with zero belongs_to and zero dilemma_impacts.
    structural_beats: list[tuple[str, str, str]] = [
        ("opening", "The hero leaves the village on a quest.", "setup"),
        ("climax", "The consequences of all choices converge.", "epilogue"),
    ]
    for beat_id, summary, role in structural_beats:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "role": role,
                "entities": ["character::hero"],
                "dilemma_impacts": [],
            },
        )

    # Pre-commit beats (belongs_to BOTH paths of their dilemma — Y-shape).
    # Entity sets are disjoint across the two dilemmas so that the e2e
    # fixture produces no intersection candidates (keeping the mocked
    # Phase 3 "empty intersections" response valid under R-2.3/R-2.8).
    pre_commit_beats: list[tuple[str, str, list[str], list[dict[str, str]], list[str]]] = [
        (
            "mt_encounter",
            "The mentor meets the seeker on the road.",
            mentor_path_ids,
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "reveals"}],
            ["character::mentor"],
        ),
        (
            "mt_test",
            "The mentor offers a dangerous shortcut through the caves.",
            mentor_path_ids,
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}],
            ["character::mentor"],
        ),
        (
            "aq_discovery",
            "The crystal is found in the temple ruins.",
            artifact_path_ids,
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "reveals"}],
            ["object::artifact", "location::temple"],
        ),
        (
            "aq_trial",
            "The crystal whispers promises of power.",
            artifact_path_ids,
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "advances"}],
            ["object::artifact"],
        ),
    ]
    for beat_id, summary, paths, impacts, entities in pre_commit_beats:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "entities": entities,
                "dilemma_impacts": impacts,
            },
        )
        for path_id in paths:
            graph.add_edge("belongs_to", f"beat::{beat_id}", f"path::{path_id}")

    # Commit beats (single-path belongs_to).  Like pre-commit beats, entity
    # sets stay disjoint across dilemmas so no cross-dilemma intersection
    # candidates are produced.
    commit_defs: list[tuple[str, str, str, str, list[str]]] = [
        (
            "mt_trust",
            "The seeker follows the mentor's guidance completely.",
            "mentor_trust_canonical",
            "mentor_trust",
            ["character::mentor"],
        ),
        (
            "mt_distrust",
            "The seeker rejects the mentor and walks away.",
            "mentor_trust_alt",
            "mentor_trust",
            ["character::mentor"],
        ),
        (
            "aq_wield",
            "The crystal heals the village.",
            "artifact_quest_canonical",
            "artifact_quest",
            ["object::artifact", "location::village"],
        ),
        (
            "aq_corrupt",
            "The crystal bends to personal gain.",
            "artifact_quest_alt",
            "artifact_quest",
            ["object::artifact"],
        ),
    ]
    for beat_id, summary, path_id, dilemma_id, entities in commit_defs:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "entities": entities,
                "dilemma_impacts": [{"dilemma_id": f"dilemma::{dilemma_id}", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", f"beat::{beat_id}", f"path::{path_id}")

    # Post-commit beats (2 per path, single belongs_to; R-3.12 min=2)
    for commit_id, _summary, path_id, _dilemma, entities in commit_defs:
        for i in range(1, 3):
            post_id = f"{commit_id}_post_{i:02d}"
            graph.create_node(
                f"beat::{post_id}",
                {
                    "type": "beat",
                    "raw_id": post_id,
                    "summary": f"Aftermath {i} of {commit_id}.",
                    "entities": entities,
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", f"beat::{post_id}", f"path::{path_id}")

    # Beat ordering (predecessor edges).  Structural beats bookend; each
    # dilemma has its own Y-shape chain; all paths converge at climax.
    ordering: list[tuple[str, str]] = [
        # opening → per-dilemma pre-commit chains
        ("mt_encounter", "opening"),
        ("aq_discovery", "opening"),
        # pre-commit chains
        ("mt_test", "mt_encounter"),
        ("aq_trial", "aq_discovery"),
        # pre-commit → commit (Y-fork)
        ("mt_trust", "mt_test"),
        ("mt_distrust", "mt_test"),
        ("aq_wield", "aq_trial"),
        ("aq_corrupt", "aq_trial"),
        # commit → post-commit chain
        ("mt_trust_post_01", "mt_trust"),
        ("mt_trust_post_02", "mt_trust_post_01"),
        ("mt_distrust_post_01", "mt_distrust"),
        ("mt_distrust_post_02", "mt_distrust_post_01"),
        ("aq_wield_post_01", "aq_wield"),
        ("aq_wield_post_02", "aq_wield_post_01"),
        ("aq_corrupt_post_01", "aq_corrupt"),
        ("aq_corrupt_post_02", "aq_corrupt_post_01"),
        # post-commit → climax (convergence)
        ("climax", "mt_trust_post_02"),
        ("climax", "mt_distrust_post_02"),
        ("climax", "aq_wield_post_02"),
        ("climax", "aq_corrupt_post_02"),
    ]
    for from_beat, to_beat in ordering:
        graph.add_edge("predecessor", f"beat::{from_beat}", f"beat::{to_beat}")

    # Consequences (R-3.4 ripples).  One per path.
    for cons_id, path_id, desc, ripples in [
        (
            "mentor_trusted",
            "mentor_trust_canonical",
            "The mentor becomes a loyal ally.",
            ["mentor aids later struggles"],
        ),
        (
            "mentor_distrusted",
            "mentor_trust_alt",
            "The mentor becomes a bitter enemy.",
            ["mentor obstructs later struggles"],
        ),
        (
            "artifact_saved",
            "artifact_quest_canonical",
            "The village is healed.",
            ["village prospers", "hero becomes a hero of legend"],
        ),
        (
            "artifact_corrupted",
            "artifact_quest_alt",
            "The hero gains dark power.",
            ["hero becomes feared", "village suffers"],
        ),
    ]:
        graph.create_node(
            f"consequence::{cons_id}",
            {
                "type": "consequence",
                "raw_id": cons_id,
                "path_id": path_id,
                "description": desc,
                "ripples": ripples,
            },
        )
        graph.add_edge("has_consequence", f"path::{path_id}", f"consequence::{cons_id}")

    # Path Freeze approval (R-6.4)
    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": True})

    graph.set_last_stage("seed")
    return graph


def make_intersection_candidate_graph() -> Graph:
    """Create a graph with beats from different dilemmas that share locations.

    Structure:
        dilemma: mentor_trust (2 paths)
        dilemma: artifact_quest (2 paths)

    Key beats for intersection testing:
        beat::mentor_meet: location="market", mentor_trust paths
        beat::artifact_discover: location="docks", flexibility edge to "market",
                                 artifact_quest paths

    These beats share "market" as a location signal and are from different
    dilemmas, making them valid intersection candidates.

    Returns:
        Graph with location-overlap intersection candidates.
    """
    graph = make_two_dilemma_graph()

    # Create location entity nodes referenced by beats
    graph.create_node(
        "location::market", {"type": "entity", "raw_id": "market", "category": "location"}
    )
    graph.create_node(
        "location::docks", {"type": "entity", "raw_id": "docks", "category": "location"}
    )

    # Add location data to some beats for intersection detection
    graph.update_node("beat::mentor_meet", location="location::market")
    graph.update_node("beat::artifact_discover", location="location::docks")
    # Flexibility edge (Story Graph Ontology): beat can also occur at market
    graph.add_edge("flexibility", "beat::artifact_discover", "location::market", role="location")

    return graph


def make_conditional_prerequisite_graph() -> Graph:
    """Create a graph where an intersection candidate has a path-specific prerequisite.

    Extends ``make_intersection_candidate_graph`` by adding a gap beat that
    belongs to only one path and is required by one of the intersection
    candidates.

    Structure:
        beat::gap_1: belongs to path::mentor_trust_canonical only
        beat::mentor_meet requires beat::gap_1

    This creates a conditional prerequisite: if mentor_meet and
    artifact_discover are proposed as an intersection, the intersection
    would span all 4 paths but gap_1 exists on only 1 path.  The
    ``check_intersection_compatibility`` invariant should reject this.

    Returns:
        Graph with a path-specific prerequisite on an intersection candidate.
    """
    graph = make_intersection_candidate_graph()

    # Add a gap beat belonging to only one path
    graph.create_node(
        "beat::gap_1",
        {
            "type": "beat",
            "raw_id": "gap_1",
            "summary": "A transition gap beat.",
            "scene_type": "sequel",
            "paths": ["mentor_trust_canonical"],
            "is_gap_beat": True,
        },
    )
    graph.add_edge("belongs_to", "beat::gap_1", "path::mentor_trust_canonical")

    # mentor_meet requires gap_1 (gap_1 must come first)
    graph.add_edge("predecessor", "beat::mentor_meet", "beat::gap_1")

    return graph
