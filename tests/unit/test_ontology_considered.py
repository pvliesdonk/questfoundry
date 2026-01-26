"""Tests for the 'explored' to 'considered' ontology change.

This module tests:
1. Backward compatibility: old 'explored' field migrates to 'considered'
2. Development state derivation from thread existence
3. Arc count derivation from threads (not from considered field)
4. Tension immutability during pruning
5. Scoped ID standardization in pruning (issue #219, PR #220)
"""

from __future__ import annotations

from questfoundry.graph.context import (
    STATE_COMMITTED,
    STATE_DEFERRED,
    STATE_LATENT,
    count_threads_per_tension,
    get_tension_development_states,
)
from questfoundry.graph.seed_pruning import (
    _prune_demoted_tensions,
    compute_arc_count,
    prune_to_arc_limit,
)
from questfoundry.models.seed import (
    Consequence,
    InitialBeat,
    SeedOutput,
    TensionDecision,
    Thread,
)


class TestTensionDecisionBackwardCompat:
    """Test backward compatibility for old 'explored' field."""

    def test_new_considered_field_works(self) -> None:
        """New 'considered' field is accepted."""
        decision = TensionDecision(
            tension_id="test_tension",
            considered=["alt_a", "alt_b"],
            implicit=["alt_c"],
        )
        assert decision.considered == ["alt_a", "alt_b"]
        assert decision.implicit == ["alt_c"]

    def test_old_explored_field_migrates(self) -> None:
        """Old 'explored' field is migrated to 'considered'."""
        # Simulate old graph data with 'explored' field
        old_data = {
            "tension_id": "test_tension",
            "explored": ["alt_a", "alt_b"],
            "implicit": ["alt_c"],
        }
        decision = TensionDecision.model_validate(old_data)
        assert decision.considered == ["alt_a", "alt_b"]
        assert decision.implicit == ["alt_c"]

    def test_considered_takes_precedence_over_explored(self) -> None:
        """If both 'considered' and 'explored' present, 'considered' is used."""
        # This shouldn't happen in practice, but test the behavior
        data = {
            "tension_id": "test_tension",
            "considered": ["new_alt"],
            "explored": ["old_alt"],
            "implicit": [],
        }
        decision = TensionDecision.model_validate(data)
        assert decision.considered == ["new_alt"]


class TestCountThreadsPerTension:
    """Test counting threads per tension."""

    def test_counts_threads_correctly(self) -> None:
        """Counts threads grouped by tension_id."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
                TensionDecision(tension_id="t2", considered=["c"], implicit=[]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="t1",
                    alternative_id="a",
                    thread_importance="major",
                    description="desc",
                ),
                Thread(
                    thread_id="th2",
                    name="Thread 2",
                    tension_id="t1",
                    alternative_id="b",
                    thread_importance="major",
                    description="desc",
                ),
                Thread(
                    thread_id="th3",
                    name="Thread 3",
                    tension_id="t2",
                    alternative_id="c",
                    thread_importance="minor",
                    description="desc",
                ),
            ],
        )

        counts = count_threads_per_tension(seed)

        assert counts == {"t1": 2, "t2": 1}

    def test_handles_scoped_tension_ids(self) -> None:
        """Handles tension_id with scope prefix."""
        seed = SeedOutput(
            tensions=[TensionDecision(tension_id="t1", considered=["a"], implicit=[])],
            threads=[
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="tension::t1",  # Scoped
                    alternative_id="a",
                    thread_importance="major",
                    description="desc",
                ),
            ],
        )

        counts = count_threads_per_tension(seed)

        assert counts == {"t1": 1}

    def test_empty_threads_returns_empty_dict(self) -> None:
        """Returns empty dict when no threads."""
        seed = SeedOutput(
            tensions=[TensionDecision(tension_id="t1", considered=["a"], implicit=[])],
            threads=[],
        )

        counts = count_threads_per_tension(seed)

        assert counts == {}


class TestGetTensionDevelopmentStates:
    """Test derivation of development states from thread existence."""

    def test_committed_state_when_thread_exists(self) -> None:
        """Alternative in 'considered' with thread is 'committed'."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="t1",
                    alternative_id="a",
                    thread_importance="major",
                    description="desc",
                ),
                Thread(
                    thread_id="th2",
                    name="Thread 2",
                    tension_id="t1",
                    alternative_id="b",
                    thread_importance="major",
                    description="desc",
                ),
            ],
        )

        states = get_tension_development_states(seed)

        assert states["t1"]["a"] == STATE_COMMITTED
        assert states["t1"]["b"] == STATE_COMMITTED

    def test_deferred_state_when_considered_but_no_thread(self) -> None:
        """Alternative in 'considered' without thread is 'deferred'."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="t1",
                    alternative_id="a",  # Only 'a' has thread
                    thread_importance="major",
                    description="desc",
                ),
            ],
        )

        states = get_tension_development_states(seed)

        assert states["t1"]["a"] == STATE_COMMITTED
        assert states["t1"]["b"] == STATE_DEFERRED  # Considered but no thread

    def test_latent_state_for_implicit_alternatives(self) -> None:
        """Alternative in 'implicit' is 'latent'."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a"], implicit=["b"]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="t1",
                    alternative_id="a",
                    thread_importance="major",
                    description="desc",
                ),
            ],
        )

        states = get_tension_development_states(seed)

        assert states["t1"]["a"] == STATE_COMMITTED
        assert states["t1"]["b"] == STATE_LATENT  # In implicit, never considered


class TestComputeArcCountFromThreads:
    """Test that arc count is derived from thread existence, not considered field."""

    def test_arc_count_from_threads_not_considered(self) -> None:
        """Arc count is 2^n where n = tensions with 2+ threads."""
        seed = SeedOutput(
            tensions=[
                # Both alts considered, but only one has thread
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
                # Both alts considered and have threads
                TensionDecision(tension_id="t2", considered=["c", "d"], implicit=[]),
            ],
            threads=[
                # t1: only one thread (despite 2 considered)
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="t1",
                    alternative_id="a",
                    thread_importance="major",
                    description="desc",
                ),
                # t2: two threads
                Thread(
                    thread_id="th2",
                    name="Thread 2",
                    tension_id="t2",
                    alternative_id="c",
                    thread_importance="major",
                    description="desc",
                ),
                Thread(
                    thread_id="th3",
                    name="Thread 3",
                    tension_id="t2",
                    alternative_id="d",
                    thread_importance="major",
                    description="desc",
                ),
            ],
        )

        arc_count = compute_arc_count(seed)

        # Only t2 has 2+ threads, so 2^1 = 2 arcs
        assert arc_count == 2

    def test_arc_count_one_when_no_fully_developed(self) -> None:
        """Arc count is 1 when no tensions have 2+ threads."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    name="Thread 1",
                    tension_id="t1",
                    alternative_id="a",
                    thread_importance="major",
                    description="desc",
                ),
            ],
        )

        arc_count = compute_arc_count(seed)

        assert arc_count == 1

    def test_arc_count_multiple_tensions(self) -> None:
        """Arc count is 2^n for multiple fully developed tensions."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
                TensionDecision(tension_id="t2", considered=["c", "d"], implicit=[]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    tension_id="t1",
                    alternative_id="a",
                    name="T1",
                    thread_importance="major",
                    description="d",
                ),
                Thread(
                    thread_id="th2",
                    tension_id="t1",
                    alternative_id="b",
                    name="T2",
                    thread_importance="major",
                    description="d",
                ),
                Thread(
                    thread_id="th3",
                    tension_id="t2",
                    alternative_id="c",
                    name="T3",
                    thread_importance="major",
                    description="d",
                ),
                Thread(
                    thread_id="th4",
                    tension_id="t2",
                    alternative_id="d",
                    name="T4",
                    thread_importance="major",
                    description="d",
                ),
            ],
        )

        arc_count = compute_arc_count(seed)

        # Both tensions have 2 threads: 2^2 = 4 arcs
        assert arc_count == 4


class TestPruningImmutability:
    """Test that pruning doesn't mutate tension decisions."""

    def _make_seed_output_with_threads(
        self,
        tensions_count: int = 5,
        threads_per_tension: int = 2,
    ) -> SeedOutput:
        """Helper to create a SeedOutput with multiple fully developed tensions."""
        tensions = []
        threads = []
        beats = []

        for i in range(tensions_count):
            tid = f"t{i}"
            tensions.append(
                TensionDecision(
                    tension_id=tid,
                    considered=["a", "b"],
                    implicit=[],
                )
            )

            for j in range(threads_per_tension):
                alt_id = "a" if j == 0 else "b"
                thread_id = f"th_{tid}_{alt_id}"
                threads.append(
                    Thread(
                        thread_id=thread_id,
                        name=f"Thread {tid} {alt_id}",
                        tension_id=tid,
                        alternative_id=alt_id,
                        thread_importance="major",
                        description="desc",
                    )
                )

                # Add a commits beat for each thread
                beats.append(
                    InitialBeat(
                        beat_id=f"beat_{thread_id}",
                        summary="Summary",
                        threads=[thread_id],
                        tension_impacts=[
                            {"tension_id": tid, "effect": "commits", "note": "Commits the tension"}
                        ],
                    )
                )

        return SeedOutput(
            tensions=tensions,
            threads=threads,
            initial_beats=beats,
        )

    def test_pruning_does_not_mutate_considered_field(self) -> None:
        """Pruning drops threads but doesn't change tension.considered."""
        seed = self._make_seed_output_with_threads(tensions_count=6, threads_per_tension=2)

        # All tensions have considered=["a", "b"]
        original_considered = {t.tension_id: list(t.considered) for t in seed.tensions}

        # Prune to max 4 arcs (2 fully developed tensions)
        pruned = prune_to_arc_limit(seed, max_arcs=4)

        # Verify tensions were not mutated
        for tension in pruned.tensions:
            assert tension.considered == original_considered[tension.tension_id], (
                f"Tension {tension.tension_id} was mutated: "
                f"expected {original_considered[tension.tension_id]}, got {tension.considered}"
            )

    def test_pruning_reduces_thread_count(self) -> None:
        """Pruning removes threads to stay within arc limit."""
        seed = self._make_seed_output_with_threads(tensions_count=6, threads_per_tension=2)

        # 6 tensions * 2 threads = 2^6 = 64 arcs before pruning
        assert compute_arc_count(seed) == 64

        # Prune to max 4 arcs (2 fully developed tensions)
        pruned = prune_to_arc_limit(seed, max_arcs=4)

        # After pruning: 2 tensions with 2 threads = 2^2 = 4 arcs
        assert compute_arc_count(pruned) <= 4

        # Threads were reduced
        assert len(pruned.threads) < len(seed.threads)

    def test_pruning_keeps_all_tensions(self) -> None:
        """Pruning keeps all tension decisions (just drops threads)."""
        seed = self._make_seed_output_with_threads(tensions_count=6, threads_per_tension=2)

        pruned = prune_to_arc_limit(seed, max_arcs=4)

        # All tension decisions are preserved
        assert len(pruned.tensions) == len(seed.tensions)
        original_ids = {t.tension_id for t in seed.tensions}
        pruned_ids = {t.tension_id for t in pruned.tensions}
        assert pruned_ids == original_ids


class TestScopedIdStandardization:
    """Tests for scoped ID handling in pruning.

    Verifies that pruning works correctly when IDs use different formats:
    - Raw IDs: `mira_fabrication`
    - Scoped IDs: `thread::mira_fabrication`

    This addresses the bug where beat.threads contained scoped IDs but
    threads_to_drop contained raw IDs, causing comparison to fail.
    See issue #219 and PR #220 for context.
    """

    def test_pruning_handles_scoped_thread_ids_in_beats(self) -> None:
        """Pruning correctly drops threads when beats use scoped IDs."""
        # Create seed with scoped thread IDs in beats
        seed = SeedOutput(
            tensions=[
                TensionDecision(
                    tension_id="tension::artifact_origin",
                    considered=["natural", "crafted"],
                    implicit=[],
                ),
            ],
            threads=[
                Thread(
                    thread_id="thread::artifact_natural",
                    name="Natural Origin",
                    tension_id="tension::artifact_origin",
                    alternative_id="natural",  # Canonical (first in considered)
                    thread_importance="major",
                    description="Artifact formed naturally",
                ),
                Thread(
                    thread_id="thread::artifact_crafted",
                    name="Crafted Origin",
                    tension_id="tension::artifact_origin",
                    alternative_id="crafted",  # Non-canonical
                    thread_importance="major",
                    description="Artifact was crafted",
                ),
            ],
            initial_beats=[
                InitialBeat(
                    beat_id="artifact_beat_01",
                    summary="Discovery of the artifact",
                    threads=["thread::artifact_natural", "thread::artifact_crafted"],  # Scoped!
                    tension_impacts=[
                        {
                            "tension_id": "tension::artifact_origin",
                            "effect": "commits",
                            "note": "Commits the tension",
                        }
                    ],
                ),
                InitialBeat(
                    beat_id="artifact_beat_02",
                    summary="Beat only for crafted thread",
                    threads=["thread::artifact_crafted"],  # Scoped!
                    tension_impacts=[
                        {
                            "tension_id": "tension::artifact_origin",
                            "effect": "reveals",
                            "note": "Reveals details",
                        }
                    ],
                ),
            ],
        )

        # Demote the tension - should drop non-canonical (crafted) thread
        demoted = {"tension::artifact_origin"}
        pruned = _prune_demoted_tensions(seed, demoted)

        # Non-canonical thread should be dropped
        thread_ids = [t.thread_id for t in pruned.threads]
        assert "thread::artifact_natural" in thread_ids
        assert "thread::artifact_crafted" not in thread_ids

        # Beat 2 should be dropped (only served crafted thread)
        beat_ids = [b.beat_id for b in pruned.initial_beats]
        assert "artifact_beat_01" in beat_ids  # Kept - serves natural
        assert "artifact_beat_02" not in beat_ids  # Dropped - only served crafted

        # Beat 1 should have crafted thread removed from its threads list
        beat_1 = next(b for b in pruned.initial_beats if b.beat_id == "artifact_beat_01")
        assert "thread::artifact_natural" in beat_1.threads
        assert "thread::artifact_crafted" not in beat_1.threads

    def test_pruning_handles_scoped_thread_ids_in_consequences(self) -> None:
        """Pruning correctly drops consequences when they use scoped IDs."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(
                    tension_id="t1",
                    considered=["alt_a", "alt_b"],
                    implicit=[],
                ),
            ],
            threads=[
                Thread(
                    thread_id="thread::th_a",
                    name="Thread A",
                    tension_id="t1",
                    alternative_id="alt_a",  # Canonical
                    thread_importance="major",
                    description="desc",
                ),
                Thread(
                    thread_id="thread::th_b",
                    name="Thread B",
                    tension_id="t1",
                    alternative_id="alt_b",  # Non-canonical
                    thread_importance="major",
                    description="desc",
                ),
            ],
            consequences=[
                Consequence(
                    consequence_id="cons_a",
                    thread_id="thread::th_a",  # Scoped
                    description="Consequence for thread A",
                ),
                Consequence(
                    consequence_id="cons_b",
                    thread_id="thread::th_b",  # Scoped
                    description="Consequence for thread B",
                ),
            ],
            initial_beats=[
                InitialBeat(
                    beat_id="beat_1",
                    summary="Test beat",
                    threads=["thread::th_a"],
                    tension_impacts=[{"tension_id": "t1", "effect": "commits", "note": "n"}],
                ),
            ],
        )

        # Demote t1 - should drop non-canonical (alt_b / th_b) thread and its consequence
        pruned = _prune_demoted_tensions(seed, {"t1"})

        # Thread B and its consequence should be dropped
        thread_ids = [t.thread_id for t in pruned.threads]
        consequence_ids = [c.consequence_id for c in pruned.consequences]

        assert "thread::th_a" in thread_ids
        assert "thread::th_b" not in thread_ids
        assert "cons_a" in consequence_ids
        assert "cons_b" not in consequence_ids

    def test_pruning_handles_mixed_scoped_and_raw_ids(self) -> None:
        """Pruning works when demoted IDs are raw but model IDs are scoped."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(
                    tension_id="tension::keeper_trust",  # Scoped in tension
                    considered=["protector", "manipulator"],
                    implicit=[],
                ),
            ],
            threads=[
                Thread(
                    thread_id="thread::keeper_protector",
                    name="Keeper as Protector",
                    tension_id="tension::keeper_trust",  # Scoped
                    alternative_id="protector",  # Canonical
                    thread_importance="major",
                    description="desc",
                ),
                Thread(
                    thread_id="thread::keeper_manipulator",
                    name="Keeper as Manipulator",
                    tension_id="tension::keeper_trust",  # Scoped
                    alternative_id="manipulator",  # Non-canonical
                    thread_importance="minor",
                    description="desc",
                ),
            ],
            initial_beats=[
                InitialBeat(
                    beat_id="keeper_beat_1",
                    summary="Meeting the keeper",
                    threads=["thread::keeper_protector", "thread::keeper_manipulator"],
                    tension_impacts=[
                        {
                            "tension_id": "tension::keeper_trust",
                            "effect": "commits",
                            "note": "n",
                        }
                    ],
                ),
            ],
        )

        # Demote using RAW ID (as might come from tension scoring)
        pruned = _prune_demoted_tensions(seed, {"keeper_trust"})  # Raw!

        # Non-canonical thread should still be dropped
        thread_ids = [t.thread_id for t in pruned.threads]
        assert "thread::keeper_protector" in thread_ids
        assert "thread::keeper_manipulator" not in thread_ids

    def test_compute_arc_count_handles_scoped_tension_ids(self) -> None:
        """compute_arc_count groups threads correctly with scoped tension_ids."""
        seed = SeedOutput(
            tensions=[
                TensionDecision(tension_id="t1", considered=["a", "b"], implicit=[]),
            ],
            threads=[
                Thread(
                    thread_id="th1",
                    name="T1",
                    tension_id="tension::t1",  # Scoped
                    alternative_id="a",
                    thread_importance="major",
                    description="d",
                ),
                Thread(
                    thread_id="th2",
                    name="T2",
                    tension_id="tension::t1",  # Scoped
                    alternative_id="b",
                    thread_importance="major",
                    description="d",
                ),
            ],
        )

        # Should correctly count 2 threads for t1 → 2^1 = 2 arcs
        arc_count = compute_arc_count(seed)
        assert arc_count == 2
