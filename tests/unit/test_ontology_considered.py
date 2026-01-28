"""Tests for the 'explored' to 'considered' ontology change.

This module tests:
1. Backward compatibility: old 'explored' field migrates to 'considered'
2. Development state derivation from path existence
3. Arc count derivation from paths (not from considered field)
4. Dilemma immutability during pruning
5. Scoped ID standardization in pruning (issue #219, PR #220)
"""

from __future__ import annotations

from questfoundry.graph.context import (
    STATE_COMMITTED,
    STATE_DEFERRED,
    STATE_LATENT,
    count_paths_per_dilemma,
    get_dilemma_development_states,
)
from questfoundry.graph.seed_pruning import (
    _prune_demoted_dilemmas,
    compute_arc_count,
    prune_to_arc_limit,
)
from questfoundry.models.seed import (
    Consequence,
    DilemmaDecision,
    InitialBeat,
    Path,
    SeedOutput,
)


class TestDilemmaDecisionBackwardCompat:
    """Test backward compatibility for old 'explored' field."""

    def test_new_considered_field_works(self) -> None:
        """New 'considered' field is accepted."""
        decision = DilemmaDecision(
            dilemma_id="test_dilemma",
            considered=["alt_a", "alt_b"],
            implicit=["alt_c"],
        )
        assert decision.considered == ["alt_a", "alt_b"]
        assert decision.implicit == ["alt_c"]

    def test_old_explored_field_migrates(self) -> None:
        """Old 'explored' field is migrated to 'considered'."""
        # Simulate old graph data with 'explored' field
        old_data = {
            "dilemma_id": "test_dilemma",
            "explored": ["alt_a", "alt_b"],
            "implicit": ["alt_c"],
        }
        decision = DilemmaDecision.model_validate(old_data)
        assert decision.considered == ["alt_a", "alt_b"]
        assert decision.implicit == ["alt_c"]

    def test_considered_takes_precedence_over_explored(self) -> None:
        """If both 'considered' and 'explored' present, 'considered' is used."""
        # This shouldn't happen in practice, but test the behavior
        data = {
            "dilemma_id": "test_dilemma",
            "considered": ["new_alt"],
            "explored": ["old_alt"],
            "implicit": [],
        }
        decision = DilemmaDecision.model_validate(data)
        assert decision.considered == ["new_alt"]


class TestCountPathsPerDilemma:
    """Test counting paths per dilemma."""

    def test_counts_paths_correctly(self) -> None:
        """Counts paths grouped by dilemma_id."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
                DilemmaDecision(dilemma_id="t2", considered=["c"], implicit=[]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="t1",
                    answer_id="a",
                    path_importance="major",
                    description="desc",
                ),
                Path(
                    path_id="th2",
                    name="Path 2",
                    dilemma_id="t1",
                    answer_id="b",
                    path_importance="major",
                    description="desc",
                ),
                Path(
                    path_id="th3",
                    name="Path 3",
                    dilemma_id="t2",
                    answer_id="c",
                    path_importance="minor",
                    description="desc",
                ),
            ],
        )

        counts = count_paths_per_dilemma(seed)

        assert counts == {"t1": 2, "t2": 1}

    def test_handles_scoped_dilemma_ids(self) -> None:
        """Handles dilemma_id with scope prefix."""
        seed = SeedOutput(
            dilemmas=[DilemmaDecision(dilemma_id="t1", considered=["a"], implicit=[])],
            paths=[
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="dilemma::t1",  # Scoped
                    answer_id="a",
                    path_importance="major",
                    description="desc",
                ),
            ],
        )

        counts = count_paths_per_dilemma(seed)

        assert counts == {"t1": 1}

    def test_empty_paths_returns_empty_dict(self) -> None:
        """Returns empty dict when no paths."""
        seed = SeedOutput(
            dilemmas=[DilemmaDecision(dilemma_id="t1", considered=["a"], implicit=[])],
            paths=[],
        )

        counts = count_paths_per_dilemma(seed)

        assert counts == {}


class TestGetDilemmaDevelopmentStates:
    """Test derivation of development states from path existence."""

    def test_committed_state_when_path_exists(self) -> None:
        """Alternative in 'considered' with path is 'committed'."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="t1",
                    answer_id="a",
                    path_importance="major",
                    description="desc",
                ),
                Path(
                    path_id="th2",
                    name="Path 2",
                    dilemma_id="t1",
                    answer_id="b",
                    path_importance="major",
                    description="desc",
                ),
            ],
        )

        states = get_dilemma_development_states(seed)

        assert states["t1"]["a"] == STATE_COMMITTED
        assert states["t1"]["b"] == STATE_COMMITTED

    def test_deferred_state_when_considered_but_no_path(self) -> None:
        """Alternative in 'considered' without path is 'deferred'."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="t1",
                    answer_id="a",  # Only 'a' has path
                    path_importance="major",
                    description="desc",
                ),
            ],
        )

        states = get_dilemma_development_states(seed)

        assert states["t1"]["a"] == STATE_COMMITTED
        assert states["t1"]["b"] == STATE_DEFERRED  # Considered but no path

    def test_latent_state_for_implicit_alternatives(self) -> None:
        """Alternative in 'implicit' is 'latent'."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a"], implicit=["b"]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="t1",
                    answer_id="a",
                    path_importance="major",
                    description="desc",
                ),
            ],
        )

        states = get_dilemma_development_states(seed)

        assert states["t1"]["a"] == STATE_COMMITTED
        assert states["t1"]["b"] == STATE_LATENT  # In implicit, never considered


class TestComputeArcCountFromPaths:
    """Test that arc count is derived from path existence, not considered field."""

    def test_arc_count_from_paths_not_considered(self) -> None:
        """Arc count is 2^n where n = dilemmas with 2+ paths."""
        seed = SeedOutput(
            dilemmas=[
                # Both alts considered, but only one has path
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
                # Both alts considered and have paths
                DilemmaDecision(dilemma_id="t2", considered=["c", "d"], implicit=[]),
            ],
            paths=[
                # t1: only one path (despite 2 considered)
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="t1",
                    answer_id="a",
                    path_importance="major",
                    description="desc",
                ),
                # t2: two paths
                Path(
                    path_id="th2",
                    name="Path 2",
                    dilemma_id="t2",
                    answer_id="c",
                    path_importance="major",
                    description="desc",
                ),
                Path(
                    path_id="th3",
                    name="Path 3",
                    dilemma_id="t2",
                    answer_id="d",
                    path_importance="major",
                    description="desc",
                ),
            ],
        )

        arc_count = compute_arc_count(seed)

        # Only t2 has 2+ paths, so 2^1 = 2 arcs
        assert arc_count == 2

    def test_arc_count_one_when_no_fully_developed(self) -> None:
        """Arc count is 1 when no dilemmas have 2+ paths."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    name="Path 1",
                    dilemma_id="t1",
                    answer_id="a",
                    path_importance="major",
                    description="desc",
                ),
            ],
        )

        arc_count = compute_arc_count(seed)

        assert arc_count == 1

    def test_arc_count_multiple_dilemmas(self) -> None:
        """Arc count is 2^n for multiple fully developed dilemmas."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
                DilemmaDecision(dilemma_id="t2", considered=["c", "d"], implicit=[]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    dilemma_id="t1",
                    answer_id="a",
                    name="T1",
                    path_importance="major",
                    description="d",
                ),
                Path(
                    path_id="th2",
                    dilemma_id="t1",
                    answer_id="b",
                    name="T2",
                    path_importance="major",
                    description="d",
                ),
                Path(
                    path_id="th3",
                    dilemma_id="t2",
                    answer_id="c",
                    name="T3",
                    path_importance="major",
                    description="d",
                ),
                Path(
                    path_id="th4",
                    dilemma_id="t2",
                    answer_id="d",
                    name="T4",
                    path_importance="major",
                    description="d",
                ),
            ],
        )

        arc_count = compute_arc_count(seed)

        # Both dilemmas have 2 paths: 2^2 = 4 arcs
        assert arc_count == 4


class TestPruningImmutability:
    """Test that pruning doesn't mutate dilemma decisions."""

    def _make_seed_output_with_paths(
        self,
        dilemmas_count: int = 5,
        paths_per_dilemma: int = 2,
    ) -> SeedOutput:
        """Helper to create a SeedOutput with multiple fully developed dilemmas."""
        dilemmas = []
        paths = []
        beats = []

        for i in range(dilemmas_count):
            tid = f"t{i}"
            dilemmas.append(
                DilemmaDecision(
                    dilemma_id=tid,
                    considered=["a", "b"],
                    implicit=[],
                )
            )

            for j in range(paths_per_dilemma):
                alt_id = "a" if j == 0 else "b"
                path_id = f"th_{tid}_{alt_id}"
                paths.append(
                    Path(
                        path_id=path_id,
                        name=f"Path {tid} {alt_id}",
                        dilemma_id=tid,
                        answer_id=alt_id,
                        path_importance="major",
                        description="desc",
                    )
                )

                # Add a commits beat for each path
                beats.append(
                    InitialBeat(
                        beat_id=f"beat_{path_id}",
                        summary="Summary",
                        paths=[path_id],
                        dilemma_impacts=[
                            {"dilemma_id": tid, "effect": "commits", "note": "Commits the dilemma"}
                        ],
                    )
                )

        return SeedOutput(
            dilemmas=dilemmas,
            paths=paths,
            initial_beats=beats,
        )

    def test_pruning_updates_considered_for_demoted(self) -> None:
        """Pruning moves non-canonical answers to implicit for demoted dilemmas."""
        seed = self._make_seed_output_with_paths(dilemmas_count=6, paths_per_dilemma=2)

        # All dilemmas have considered=["a", "b"]
        original_considered = {t.dilemma_id: list(t.considered) for t in seed.dilemmas}

        # Prune to max 4 arcs (2 fully developed dilemmas)
        pruned = prune_to_arc_limit(seed, max_arcs=4)

        # Build path counts by dilemma to detect demotion
        path_counts: dict[str, int] = {}
        for path in pruned.paths:
            path_counts[path.dilemma_id] = path_counts.get(path.dilemma_id, 0) + 1

        for dilemma in pruned.dilemmas:
            if path_counts.get(dilemma.dilemma_id, 0) <= 1:
                assert dilemma.considered == ["a"]
                assert "b" in dilemma.implicit
            else:
                assert dilemma.considered == original_considered[dilemma.dilemma_id]

    def test_pruning_reduces_path_count(self) -> None:
        """Pruning removes paths to stay within arc limit."""
        seed = self._make_seed_output_with_paths(dilemmas_count=6, paths_per_dilemma=2)

        # 6 dilemmas * 2 paths = 2^6 = 64 arcs before pruning
        assert compute_arc_count(seed) == 64

        # Prune to max 4 arcs (2 fully developed dilemmas)
        pruned = prune_to_arc_limit(seed, max_arcs=4)

        # After pruning: 2 dilemmas with 2 paths = 2^2 = 4 arcs
        assert compute_arc_count(pruned) <= 4

        # Paths were reduced
        assert len(pruned.paths) < len(seed.paths)

    def test_pruning_keeps_all_dilemmas(self) -> None:
        """Pruning keeps all dilemma decisions (just drops paths)."""
        seed = self._make_seed_output_with_paths(dilemmas_count=6, paths_per_dilemma=2)

        pruned = prune_to_arc_limit(seed, max_arcs=4)

        # All dilemma decisions are preserved
        assert len(pruned.dilemmas) == len(seed.dilemmas)
        original_ids = {t.dilemma_id for t in seed.dilemmas}
        pruned_ids = {t.dilemma_id for t in pruned.dilemmas}
        assert pruned_ids == original_ids


class TestScopedIdStandardization:
    """Tests for scoped ID handling in pruning.

    Verifies that pruning works correctly when IDs use different formats:
    - Raw IDs: `mira_fabrication`
    - Scoped IDs: `path::mira_fabrication`

    This addresses the bug where beat.paths contained scoped IDs but
    paths_to_drop contained raw IDs, causing comparison to fail.
    See issue #219 and PR #220 for context.
    """

    def test_pruning_handles_scoped_path_ids_in_beats(self) -> None:
        """Pruning correctly drops paths when beats use scoped IDs."""
        # Create seed with scoped path IDs in beats
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(
                    dilemma_id="dilemma::artifact_origin",
                    considered=["natural", "crafted"],
                    implicit=[],
                ),
            ],
            paths=[
                Path(
                    path_id="path::artifact_natural",
                    name="Natural Origin",
                    dilemma_id="dilemma::artifact_origin",
                    answer_id="natural",  # Canonical (first in considered)
                    path_importance="major",
                    description="Artifact formed naturally",
                ),
                Path(
                    path_id="path::artifact_crafted",
                    name="Crafted Origin",
                    dilemma_id="dilemma::artifact_origin",
                    answer_id="crafted",  # Non-canonical
                    path_importance="major",
                    description="Artifact was crafted",
                ),
            ],
            initial_beats=[
                InitialBeat(
                    beat_id="artifact_beat_01",
                    summary="Discovery of the artifact",
                    paths=["path::artifact_natural", "path::artifact_crafted"],  # Scoped!
                    dilemma_impacts=[
                        {
                            "dilemma_id": "dilemma::artifact_origin",
                            "effect": "commits",
                            "note": "Commits the dilemma",
                        }
                    ],
                ),
                InitialBeat(
                    beat_id="artifact_beat_02",
                    summary="Beat only for crafted path",
                    paths=["path::artifact_crafted"],  # Scoped!
                    dilemma_impacts=[
                        {
                            "dilemma_id": "dilemma::artifact_origin",
                            "effect": "reveals",
                            "note": "Reveals details",
                        }
                    ],
                ),
            ],
        )

        # Demote the dilemma - should drop non-canonical (crafted) path
        demoted = {"dilemma::artifact_origin"}
        pruned = _prune_demoted_dilemmas(seed, demoted)

        # Non-canonical path should be dropped
        path_ids = [t.path_id for t in pruned.paths]
        assert "path::artifact_natural" in path_ids
        assert "path::artifact_crafted" not in path_ids

        # Beat 2 should be dropped (only served crafted path)
        beat_ids = [b.beat_id for b in pruned.initial_beats]
        assert "artifact_beat_01" in beat_ids  # Kept - serves natural
        assert "artifact_beat_02" not in beat_ids  # Dropped - only served crafted

        # Beat 1 should have crafted path removed from its paths list
        beat_1 = next(b for b in pruned.initial_beats if b.beat_id == "artifact_beat_01")
        assert "path::artifact_natural" in beat_1.paths
        assert "path::artifact_crafted" not in beat_1.paths
        # Demoted dilemma should keep canonical and move non-canonical to implicit
        pruned_dilemma = next(
            d for d in pruned.dilemmas if d.dilemma_id == "dilemma::artifact_origin"
        )
        assert pruned_dilemma.considered == ["natural"]
        assert "crafted" in pruned_dilemma.implicit

    def test_pruning_handles_scoped_path_ids_in_consequences(self) -> None:
        """Pruning correctly drops consequences when they use scoped IDs."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(
                    dilemma_id="t1",
                    considered=["alt_a", "alt_b"],
                    implicit=[],
                ),
            ],
            paths=[
                Path(
                    path_id="path::th_a",
                    name="Path A",
                    dilemma_id="t1",
                    answer_id="alt_a",  # Canonical
                    path_importance="major",
                    description="desc",
                ),
                Path(
                    path_id="path::th_b",
                    name="Path B",
                    dilemma_id="t1",
                    answer_id="alt_b",  # Non-canonical
                    path_importance="major",
                    description="desc",
                ),
            ],
            consequences=[
                Consequence(
                    consequence_id="cons_a",
                    path_id="path::th_a",  # Scoped
                    description="Consequence for path A",
                ),
                Consequence(
                    consequence_id="cons_b",
                    path_id="path::th_b",  # Scoped
                    description="Consequence for path B",
                ),
            ],
            initial_beats=[
                InitialBeat(
                    beat_id="beat_1",
                    summary="Test beat",
                    paths=["path::th_a"],
                    dilemma_impacts=[{"dilemma_id": "t1", "effect": "commits", "note": "n"}],
                ),
            ],
        )

        # Demote t1 - should drop non-canonical (alt_b / th_b) path and its consequence
        pruned = _prune_demoted_dilemmas(seed, {"t1"})

        # Path B and its consequence should be dropped
        path_ids = [t.path_id for t in pruned.paths]
        consequence_ids = [c.consequence_id for c in pruned.consequences]

        assert "path::th_a" in path_ids
        assert "path::th_b" not in path_ids
        assert "cons_a" in consequence_ids
        assert "cons_b" not in consequence_ids

    def test_pruning_handles_mixed_scoped_and_raw_ids(self) -> None:
        """Pruning works when demoted IDs are raw but model IDs are scoped."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(
                    dilemma_id="dilemma::keeper_trust",  # Scoped in dilemma
                    considered=["protector", "manipulator"],
                    implicit=[],
                ),
            ],
            paths=[
                Path(
                    path_id="path::keeper_protector",
                    name="Keeper as Protector",
                    dilemma_id="dilemma::keeper_trust",  # Scoped
                    answer_id="protector",  # Canonical
                    path_importance="major",
                    description="desc",
                ),
                Path(
                    path_id="path::keeper_manipulator",
                    name="Keeper as Manipulator",
                    dilemma_id="dilemma::keeper_trust",  # Scoped
                    answer_id="manipulator",  # Non-canonical
                    path_importance="minor",
                    description="desc",
                ),
            ],
            initial_beats=[
                InitialBeat(
                    beat_id="keeper_beat_1",
                    summary="Meeting the keeper",
                    paths=["path::keeper_protector", "path::keeper_manipulator"],
                    dilemma_impacts=[
                        {
                            "dilemma_id": "dilemma::keeper_trust",
                            "effect": "commits",
                            "note": "n",
                        }
                    ],
                ),
            ],
        )

        # Demote using RAW ID (as might come from dilemma scoring)
        pruned = _prune_demoted_dilemmas(seed, {"keeper_trust"})  # Raw!

        # Non-canonical path should still be dropped
        path_ids = [t.path_id for t in pruned.paths]
        assert "path::keeper_protector" in path_ids
        assert "path::keeper_manipulator" not in path_ids

    def test_compute_arc_count_handles_scoped_dilemma_ids(self) -> None:
        """compute_arc_count groups paths correctly with scoped dilemma_ids."""
        seed = SeedOutput(
            dilemmas=[
                DilemmaDecision(dilemma_id="t1", considered=["a", "b"], implicit=[]),
            ],
            paths=[
                Path(
                    path_id="th1",
                    name="T1",
                    dilemma_id="dilemma::t1",  # Scoped
                    answer_id="a",
                    path_importance="major",
                    description="d",
                ),
                Path(
                    path_id="th2",
                    name="T2",
                    dilemma_id="dilemma::t1",  # Scoped
                    answer_id="b",
                    path_importance="major",
                    description="d",
                ),
            ],
        )

        # Should correctly count 2 paths for t1 → 2^1 = 2 arcs
        arc_count = compute_arc_count(seed)
        assert arc_count == 2
