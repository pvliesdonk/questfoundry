"""Factory functions for GROW stage test graphs.

These build pre-populated Graph objects with the node/edge structures
that GROW expects from earlier stages (BRAINSTORM, SEED).

Graph structure conventions:
- Entity nodes: entity::{raw_id}
- Tension nodes: tension::{raw_id}
- Thread nodes: thread::{raw_id}
- Beat nodes: beat::{raw_id}
- Consequence nodes: consequence::{raw_id}
- Edges: belongs_to (beat→thread), explores (thread→tension),
         has_consequence (thread→consequence), requires (beat→beat)
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph


def make_single_tension_graph() -> Graph:
    """Create a minimal graph with 1 tension, 2 threads, 4 beats.

    Structure:
        tension: mentor_trust
        threads: mentor_trust_canonical (canonical), mentor_trust_alt (alternative)
        beats: opening, mentor_meet, mentor_commits_canonical, mentor_commits_alt

    Beat ordering (requires edges):
        opening → mentor_meet → mentor_commits_canonical
        opening → mentor_meet → mentor_commits_alt

    Consequences:
        mentor_trusted (thread: mentor_trust_canonical)
        mentor_distrusted (thread: mentor_trust_alt)

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

    # Tension
    graph.create_node(
        "tension::mentor_trust",
        {
            "type": "tension",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
        },
    )

    # Threads
    graph.create_node(
        "thread::mentor_trust_canonical",
        {
            "type": "thread",
            "raw_id": "mentor_trust_canonical",
            "tension_id": "mentor_trust",
            "alternative_id": "trust_yes",
            "thread_importance": "major",
            "is_canonical": True,
        },
    )
    graph.create_node(
        "thread::mentor_trust_alt",
        {
            "type": "thread",
            "raw_id": "mentor_trust_alt",
            "tension_id": "mentor_trust",
            "alternative_id": "trust_no",
            "thread_importance": "major",
            "is_canonical": False,
        },
    )

    # Thread → Tension edges
    graph.add_edge("explores", "thread::mentor_trust_canonical", "tension::mentor_trust")
    graph.add_edge("explores", "thread::mentor_trust_alt", "tension::mentor_trust")

    # Beats
    graph.create_node(
        "beat::opening",
        {
            "type": "beat",
            "raw_id": "opening",
            "summary": "The story begins.",
            "threads": ["mentor_trust_canonical", "mentor_trust_alt"],
        },
    )
    graph.create_node(
        "beat::mentor_meet",
        {
            "type": "beat",
            "raw_id": "mentor_meet",
            "summary": "Hero meets the mentor.",
            "threads": ["mentor_trust_canonical", "mentor_trust_alt"],
        },
    )
    graph.create_node(
        "beat::mentor_commits_canonical",
        {
            "type": "beat",
            "raw_id": "mentor_commits_canonical",
            "summary": "Hero trusts the mentor fully.",
            "threads": ["mentor_trust_canonical"],
            "tension_impacts": [{"tension_id": "mentor_trust", "effect": "commits"}],
        },
    )
    graph.create_node(
        "beat::mentor_commits_alt",
        {
            "type": "beat",
            "raw_id": "mentor_commits_alt",
            "summary": "Hero rejects the mentor.",
            "threads": ["mentor_trust_alt"],
            "tension_impacts": [{"tension_id": "mentor_trust", "effect": "commits"}],
        },
    )

    # Beat → Thread edges (belongs_to)
    graph.add_edge("belongs_to", "beat::opening", "thread::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::opening", "thread::mentor_trust_alt")
    graph.add_edge("belongs_to", "beat::mentor_meet", "thread::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::mentor_meet", "thread::mentor_trust_alt")
    graph.add_edge("belongs_to", "beat::mentor_commits_canonical", "thread::mentor_trust_canonical")
    graph.add_edge("belongs_to", "beat::mentor_commits_alt", "thread::mentor_trust_alt")

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
            "thread_id": "mentor_trust_canonical",
            "description": "The mentor becomes an ally.",
        },
    )
    graph.create_node(
        "consequence::mentor_distrusted",
        {
            "type": "consequence",
            "raw_id": "mentor_distrusted",
            "thread_id": "mentor_trust_alt",
            "description": "The mentor becomes an adversary.",
        },
    )
    graph.add_edge(
        "has_consequence", "thread::mentor_trust_canonical", "consequence::mentor_trusted"
    )
    graph.add_edge("has_consequence", "thread::mentor_trust_alt", "consequence::mentor_distrusted")

    return graph


def make_two_tension_graph() -> Graph:
    """Create a graph with 2 tensions, 4 threads, 8 beats.

    Structure:
        tension: mentor_trust (2 threads: canonical, alt)
        tension: artifact_quest (2 threads: canonical, alt)

    Threads:
        mentor_trust_canonical, mentor_trust_alt
        artifact_quest_canonical, artifact_quest_alt

    Beats per thread (with shared opening/closing beats):
        opening (all threads)
        mentor_meet (mentor threads)
        artifact_discover (artifact threads)
        mentor_commits_canonical (mentor_trust_canonical)
        mentor_commits_alt (mentor_trust_alt)
        artifact_commits_canonical (artifact_quest_canonical)
        artifact_commits_alt (artifact_quest_alt)
        finale (all threads)

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

    # Tensions
    graph.create_node(
        "tension::mentor_trust",
        {
            "type": "tension",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
        },
    )
    graph.create_node(
        "tension::artifact_quest",
        {
            "type": "tension",
            "raw_id": "artifact_quest",
            "question": "Does the hero use the artifact for good?",
        },
    )

    # Threads
    for thread_id, tension_id, alt_id, is_canon in [
        ("mentor_trust_canonical", "mentor_trust", "trust_yes", True),
        ("mentor_trust_alt", "mentor_trust", "trust_no", False),
        ("artifact_quest_canonical", "artifact_quest", "use_good", True),
        ("artifact_quest_alt", "artifact_quest", "use_selfish", False),
    ]:
        graph.create_node(
            f"thread::{thread_id}",
            {
                "type": "thread",
                "raw_id": thread_id,
                "tension_id": tension_id,
                "alternative_id": alt_id,
                "thread_importance": "major",
                "is_canonical": is_canon,
            },
        )
        graph.add_edge("explores", f"thread::{thread_id}", f"tension::{tension_id}")

    # Beats
    all_threads = [
        "mentor_trust_canonical",
        "mentor_trust_alt",
        "artifact_quest_canonical",
        "artifact_quest_alt",
    ]
    mentor_threads = ["mentor_trust_canonical", "mentor_trust_alt"]
    artifact_threads = ["artifact_quest_canonical", "artifact_quest_alt"]

    beats = [
        ("opening", "The story begins.", all_threads, []),
        ("mentor_meet", "Hero meets the mentor.", mentor_threads, []),
        ("artifact_discover", "Hero discovers the artifact.", artifact_threads, []),
        (
            "mentor_commits_canonical",
            "Hero trusts the mentor.",
            ["mentor_trust_canonical"],
            [{"tension_id": "mentor_trust", "effect": "commits"}],
        ),
        (
            "mentor_commits_alt",
            "Hero distrusts the mentor.",
            ["mentor_trust_alt"],
            [{"tension_id": "mentor_trust", "effect": "commits"}],
        ),
        (
            "artifact_commits_canonical",
            "Hero uses artifact for good.",
            ["artifact_quest_canonical"],
            [{"tension_id": "artifact_quest", "effect": "commits"}],
        ),
        (
            "artifact_commits_alt",
            "Hero uses artifact selfishly.",
            ["artifact_quest_alt"],
            [{"tension_id": "artifact_quest", "effect": "commits"}],
        ),
        ("finale", "The conclusion.", all_threads, []),
    ]

    for beat_id, summary, threads, impacts in beats:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "threads": threads,
                "tension_impacts": impacts,
            },
        )
        for thread_id in threads:
            graph.add_edge("belongs_to", f"beat::{beat_id}", f"thread::{thread_id}")

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
    for cons_id, thread_id, desc in [
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
                "thread_id": thread_id,
                "description": desc,
            },
        )
        graph.add_edge("has_consequence", f"thread::{thread_id}", f"consequence::{cons_id}")

    return graph


def make_e2e_fixture_graph() -> Graph:
    """Create a detailed graph for E2E integration testing.

    Structure: 2 tensions x 2 threads = 4 arcs, 10 unique beats with proper
    lifecycle progression (introduces -> reveals -> advances -> commits).

    Tensions:
        mentor_trust: Does the hero trust the mentor?
        artifact_quest: Does the hero use the artifact for good?

    Threads (4 arcs):
        mentor_trust_canonical: hero trusts the mentor
        mentor_trust_alt: hero distrusts the mentor
        artifact_quest_canonical: hero uses artifact for good
        artifact_quest_alt: hero uses artifact selfishly

    Beats per thread (5 each, with shared beats):
        mentor_trust_canonical: opening → mt_encounter → mt_test → mt_trust → climax
        mentor_trust_alt: opening → mt_encounter → mt_test → mt_distrust → climax
        artifact_quest_canonical: opening → aq_discovery → aq_trial → aq_wield → climax
        artifact_quest_alt: opening → aq_discovery → aq_trial → aq_corrupt → climax

    Total unique beats: 10 (3 shared + 2 mentor-shared + 2 artifact-shared + 4 unique)

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

    # Tensions
    graph.create_node(
        "tension::mentor_trust",
        {
            "type": "tension",
            "raw_id": "mentor_trust",
            "question": "Does the hero trust the mentor?",
        },
    )
    graph.create_node(
        "tension::artifact_quest",
        {
            "type": "tension",
            "raw_id": "artifact_quest",
            "question": "Does the hero use the artifact for good or selfish ends?",
        },
    )

    # Threads
    thread_defs = [
        ("mentor_trust_canonical", "mentor_trust", "trust_yes", True),
        ("mentor_trust_alt", "mentor_trust", "trust_no", False),
        ("artifact_quest_canonical", "artifact_quest", "use_good", True),
        ("artifact_quest_alt", "artifact_quest", "use_selfish", False),
    ]
    all_thread_ids = [t[0] for t in thread_defs]
    mentor_thread_ids = [t[0] for t in thread_defs if "mentor" in t[0]]
    artifact_thread_ids = [t[0] for t in thread_defs if "artifact" in t[0]]

    for thread_id, tension_id, alt_id, is_canon in thread_defs:
        graph.create_node(
            f"thread::{thread_id}",
            {
                "type": "thread",
                "raw_id": thread_id,
                "tension_id": tension_id,
                "alternative_id": alt_id,
                "thread_importance": "major",
                "is_canonical": is_canon,
            },
        )
        graph.add_edge("explores", f"thread::{thread_id}", f"tension::{tension_id}")

    # Beats with lifecycle effects
    beat_defs: list[tuple[str, str, list[str], list[dict[str, str]]]] = [
        # (beat_id, summary, thread_ids, tension_impacts)
        ("opening", "The hero leaves the village on a quest.", all_thread_ids, []),
        (
            "mt_encounter",
            "The hero meets a mysterious sage on the road.",
            mentor_thread_ids,
            [{"tension_id": "mentor_trust", "effect": "reveals"}],
        ),
        (
            "mt_test",
            "The mentor offers a dangerous shortcut through the caves.",
            mentor_thread_ids,
            [{"tension_id": "mentor_trust", "effect": "advances"}],
        ),
        (
            "mt_trust",
            "The hero follows the mentor's guidance completely.",
            ["mentor_trust_canonical"],
            [{"tension_id": "mentor_trust", "effect": "commits"}],
        ),
        (
            "mt_distrust",
            "The hero rejects the mentor and goes alone.",
            ["mentor_trust_alt"],
            [{"tension_id": "mentor_trust", "effect": "commits"}],
        ),
        (
            "aq_discovery",
            "The hero finds the crystal in the temple ruins.",
            artifact_thread_ids,
            [{"tension_id": "artifact_quest", "effect": "reveals"}],
        ),
        (
            "aq_trial",
            "The crystal whispers promises of power to the hero.",
            artifact_thread_ids,
            [{"tension_id": "artifact_quest", "effect": "advances"}],
        ),
        (
            "aq_wield",
            "The hero channels the crystal to heal the village.",
            ["artifact_quest_canonical"],
            [{"tension_id": "artifact_quest", "effect": "commits"}],
        ),
        (
            "aq_corrupt",
            "The hero uses the crystal for personal gain.",
            ["artifact_quest_alt"],
            [{"tension_id": "artifact_quest", "effect": "commits"}],
        ),
        ("climax", "The consequences of all choices converge.", all_thread_ids, []),
    ]

    for beat_id, summary, threads, impacts in beat_defs:
        graph.create_node(
            f"beat::{beat_id}",
            {
                "type": "beat",
                "raw_id": beat_id,
                "summary": summary,
                "threads": threads,
                "tension_impacts": impacts,
            },
        )
        for thread_id in threads:
            graph.add_edge("belongs_to", f"beat::{beat_id}", f"thread::{thread_id}")

    # Beat ordering (requires edges)
    ordering = [
        # opening → tension-specific beats
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
    for cons_id, thread_id, desc in [
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
                "thread_id": thread_id,
                "description": desc,
            },
        )
        graph.add_edge("has_consequence", f"thread::{thread_id}", f"consequence::{cons_id}")

    return graph


def make_knot_candidate_graph() -> Graph:
    """Create a graph with beats from different tensions that share locations.

    Structure:
        tension: mentor_trust (2 threads)
        tension: artifact_quest (2 threads)

    Key beats for knot testing:
        beat::mentor_meet: location="market", mentor_trust threads
        beat::artifact_discover: location="docks", location_alternatives=["market"],
                                 artifact_quest threads

    These beats share "market" as a location signal and are from different
    tensions, making them valid knot candidates.

    Returns:
        Graph with location-overlap knot candidates.
    """
    graph = make_two_tension_graph()

    # Add location data to some beats for knot detection
    graph.update_node("beat::mentor_meet", location="market")
    graph.update_node(
        "beat::artifact_discover",
        location="docks",
        location_alternatives=["market"],
    )

    return graph
