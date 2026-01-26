"""Tests for the 'explored' to 'considered' ontology change.

This module tests:
1. Backward compatibility: old 'explored' field migrates to 'considered'
2. Development state derivation from thread existence
3. Arc count derivation from threads (not from considered field)
4. Tension immutability during pruning
"""

from __future__ import annotations

from questfoundry.graph.context import (
    STATE_COMMITTED,
    STATE_DEFERRED,
    STATE_LATENT,
    count_threads_per_tension,
    get_tension_development_states,
)
from questfoundry.graph.seed_pruning import compute_arc_count, prune_to_arc_limit
from questfoundry.models.seed import (
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
