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
        "entity::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_category": "character",
            "concept": "A wise mentor",
        },
    )
    graph.create_node(
        "entity::hero",
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
    graph.add_edge("requires", "beat::mentor_meet", "beat::opening")
    graph.add_edge("requires", "beat::mentor_commits_canonical", "beat::mentor_meet")
    graph.add_edge("requires", "beat::mentor_commits_alt", "beat::mentor_meet")

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
        "entity::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_category": "character",
            "concept": "A wise mentor",
        },
    )
    graph.create_node(
        "entity::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_category": "character",
            "concept": "The protagonist",
        },
    )
    graph.create_node(
        "entity::artifact",
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
    graph.add_edge("requires", "beat::mentor_meet", "beat::opening")
    graph.add_edge("requires", "beat::artifact_discover", "beat::opening")
    # mentor_meet → mentor_commits_*
    graph.add_edge("requires", "beat::mentor_commits_canonical", "beat::mentor_meet")
    graph.add_edge("requires", "beat::mentor_commits_alt", "beat::mentor_meet")
    # artifact_discover → artifact_commits_*
    graph.add_edge("requires", "beat::artifact_commits_canonical", "beat::artifact_discover")
    graph.add_edge("requires", "beat::artifact_commits_alt", "beat::artifact_discover")
    # commits → finale
    graph.add_edge("requires", "beat::finale", "beat::mentor_commits_canonical")
    graph.add_edge("requires", "beat::finale", "beat::mentor_commits_alt")
    graph.add_edge("requires", "beat::finale", "beat::artifact_commits_canonical")
    graph.add_edge("requires", "beat::finale", "beat::artifact_commits_alt")

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

    Structure: 2 dilemmas x 2 paths = 4 arcs, 10 unique beats with proper
    lifecycle progression (introduces -> reveals -> advances -> commits).

    Dilemmas:
        mentor_trust: Does the hero trust the mentor?
        artifact_quest: Does the hero use the artifact for good?

    Paths (4 arcs):
        mentor_trust_canonical: hero trusts the mentor
        mentor_trust_alt: hero distrusts the mentor
        artifact_quest_canonical: hero uses artifact for good
        artifact_quest_alt: hero uses artifact selfishly

    Beats per path (5 each, with shared beats):
        mentor_trust_canonical: opening → mt_encounter → mt_test → mt_trust → climax
        mentor_trust_alt: opening → mt_encounter → mt_test → mt_distrust → climax
        artifact_quest_canonical: opening → aq_discovery → aq_trial → aq_wield → climax
        artifact_quest_alt: opening → aq_discovery → aq_trial → aq_corrupt → climax

    Total unique beats: 10 (3 shared + 2 mentor-shared + 2 artifact-shared + 4 unique arcs)

    Returns:
        Populated Graph instance ready for GROW processing.
    """
    graph = Graph.empty()

    # Entities
    for eid, cat, concept in [
        ("hero", "character", "A young adventurer seeking purpose"),
        ("mentor", "character", "A mysterious sage with hidden agenda"),
        ("artifact", "object", "An ancient crystal of immense power"),
        ("temple", "location", "Crumbling temple in the mountains"),
        ("village", "location", "The hero's home village"),
    ]:
        graph.create_node(
            f"entity::{eid}",
            {
                "type": "entity",
                "raw_id": eid,
                "entity_category": cat,
                "concept": concept,
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
            "question": "Does the hero use the artifact for good or selfish ends?",
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
    path_defs = [
        ("mentor_trust_canonical", "mentor_trust", "trust_yes", True),
        ("mentor_trust_alt", "mentor_trust", "trust_no", False),
        ("artifact_quest_canonical", "artifact_quest", "use_good", True),
        ("artifact_quest_alt", "artifact_quest", "use_selfish", False),
    ]
    all_path_ids = [t[0] for t in path_defs]
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

    # Beats with lifecycle effects
    beat_defs: list[tuple[str, str, list[str], list[dict[str, str]]]] = [
        # (beat_id, summary, path_ids, dilemma_impacts)
        ("opening", "The hero leaves the village on a quest.", all_path_ids, []),
        (
            "mt_encounter",
            "The hero meets a mysterious sage on the road.",
            mentor_path_ids,
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "reveals"}],
        ),
        (
            "mt_test",
            "The mentor offers a dangerous shortcut through the caves.",
            mentor_path_ids,
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}],
        ),
        (
            "mt_trust",
            "The hero follows the mentor's guidance completely.",
            ["mentor_trust_canonical"],
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        ),
        (
            "mt_distrust",
            "The hero rejects the mentor and goes alone.",
            ["mentor_trust_alt"],
            [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        ),
        (
            "aq_discovery",
            "The hero finds the crystal in the temple ruins.",
            artifact_path_ids,
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "reveals"}],
        ),
        (
            "aq_trial",
            "The crystal whispers promises of power to the hero.",
            artifact_path_ids,
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "advances"}],
        ),
        (
            "aq_wield",
            "The hero channels the crystal to heal the village.",
            ["artifact_quest_canonical"],
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "commits"}],
        ),
        (
            "aq_corrupt",
            "The hero uses the crystal for personal gain.",
            ["artifact_quest_alt"],
            [{"dilemma_id": "dilemma::artifact_quest", "effect": "commits"}],
        ),
        ("climax", "The consequences of all choices converge.", all_path_ids, []),
    ]

    for beat_id, summary, paths, impacts in beat_defs:
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
    ordering = [
        # opening → dilemma-specific beats
        ("mt_encounter", "opening"),
        ("aq_discovery", "opening"),
        # reveals → advances
        ("mt_test", "mt_encounter"),
        ("aq_trial", "aq_discovery"),
        # advances → commits
        ("mt_trust", "mt_test"),
        ("mt_distrust", "mt_test"),
        ("aq_wield", "aq_trial"),
        ("aq_corrupt", "aq_trial"),
        # commits → climax
        ("climax", "mt_trust"),
        ("climax", "mt_distrust"),
        ("climax", "aq_wield"),
        ("climax", "aq_corrupt"),
    ]
    for from_beat, to_beat in ordering:
        graph.add_edge("requires", f"beat::{from_beat}", f"beat::{to_beat}")

    # Consequences
    for cons_id, path_id, desc in [
        ("mentor_trusted", "mentor_trust_canonical", "The mentor becomes a loyal ally."),
        ("mentor_distrusted", "mentor_trust_alt", "The mentor becomes a bitter enemy."),
        ("artifact_saved", "artifact_quest_canonical", "The village is healed."),
        ("artifact_corrupted", "artifact_quest_alt", "The hero gains dark power."),
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


def make_intersection_candidate_graph() -> Graph:
    """Create a graph with beats from different dilemmas that share locations.

    Structure:
        dilemma: mentor_trust (2 paths)
        dilemma: artifact_quest (2 paths)

    Key beats for intersection testing:
        beat::mentor_meet: location="market", mentor_trust paths
        beat::artifact_discover: location="docks", location_alternatives=["market"],
                                 artifact_quest paths

    These beats share "market" as a location signal and are from different
    dilemmas, making them valid intersection candidates.

    Returns:
        Graph with location-overlap intersection candidates.
    """
    graph = make_two_dilemma_graph()

    # Add location data to some beats for intersection detection
    graph.update_node("beat::mentor_meet", location="market")
    graph.update_node(
        "beat::artifact_discover",
        location="docks",
        location_alternatives=["market"],
    )

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
    graph.add_edge("requires", "beat::mentor_meet", "beat::gap_1")

    return graph
