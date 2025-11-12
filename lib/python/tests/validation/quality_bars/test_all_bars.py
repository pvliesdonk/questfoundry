"""Comprehensive tests for all quality bars."""

import pytest

from questfoundry.models.artifact import Artifact
from questfoundry.validation.quality_bars import (
    DeterminismBar,
    GatewaysBar,
    IntegrityBar,
    NonlinearityBar,
    PresentationBar,
    ReachabilityBar,
    SpoilerHygieneBar,
    StyleBar,
)


class TestIntegrityBar:
    """Tests for Integrity quality bar."""

    def test_integrity_pass_valid_artifact(self):
        """Test integrity passes for valid artifact."""
        bar = IntegrityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "section-1",
                    "text": "Content",
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True
        assert result.bar_name == "integrity"

    def test_integrity_fail_missing_id(self):
        """Test integrity fails for missing ID."""
        bar = IntegrityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "text": "Content without ID",
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is False
        assert len(result.blockers) >= 1

    def test_integrity_fail_dangling_reference(self):
        """Test integrity fails for dangling choice target."""
        bar = IntegrityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "section-1",
                    "text": "Choose",
                    "choices": [{"text": "Go forward", "target": "section-99"}],
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is False
        assert any("section-99" in issue.message for issue in result.blockers)

    def test_integrity_warn_unmarked_dead_end(self):
        """Test integrity warns for unmarked dead end."""
        bar = IntegrityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "section-1",
                    "text": "The end",
                    "choices": [],
                },
            )
        ]

        result = bar.validate(artifacts)
        # Should have warning about unmarked terminal
        assert len(result.warnings) >= 1


class TestReachabilityBar:
    """Tests for Reachability quality bar."""

    def test_reachability_pass_linear_path(self):
        """Test reachability passes for simple linear path."""
        bar = ReachabilityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "start",
                    "start": True,
                    "text": "Beginning",
                    "choices": [{"text": "Continue", "target": "end"}],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "end",
                    "text": "The end",
                    "terminal": True,
                },
            ),
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_reachability_fail_unreachable_keystone(self):
        """Test reachability fails for unreachable keystone."""
        bar = ReachabilityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "start",
                    "start": True,
                    "text": "Beginning",
                    "choices": [{"text": "End", "target": "end"}],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "keystone",
                    "keystone": True,
                    "text": "Important beat",
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "end",
                    "text": "The end",
                    "terminal": True,
                },
            ),
        ]

        result = bar.validate(artifacts)
        assert result.passed is False
        assert any("keystone" in issue.message.lower() for issue in result.blockers)

    def test_reachability_warn_orphaned_section(self):
        """Test reachability warns for orphaned section."""
        bar = ReachabilityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "start",
                    "start": True,
                    "text": "Beginning",
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "orphan",
                    "text": "Unreachable",
                },
            ),
        ]

        result = bar.validate(artifacts)
        # Should warn about orphaned section
        assert len(result.warnings) >= 1


class TestStyleBar:
    """Tests for Style quality bar."""

    def test_style_pass_consistent_voice(self):
        """Test style passes for consistent voice."""
        bar = StyleBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "You walked down the dark corridor.",
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s2",
                    "text": "You reached for the door handle.",
                },
            ),
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_style_warn_mixed_tense(self):
        """Test style warns for mixed tense."""
        bar = StyleBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "You walked down the corridor and you see a door.",
                },
            )
        ]

        result = bar.validate(artifacts)
        # Should warn about mixed tense
        assert len(result.warnings) >= 1


class TestGatewaysBar:
    """Tests for Gateways quality bar."""

    def test_gateways_pass_diegetic_condition(self):
        """Test gateways passes for diegetic conditions."""
        bar = GatewaysBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "A door",
                    "choices": [
                        {
                            "text": "Enter",
                            "target": "s2",
                            "requires": "union_token",
                        }
                    ],
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_gateways_warn_non_diegetic(self):
        """Test gateways warns for non-diegetic conditions."""
        bar = GatewaysBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "choices": [
                        {
                            "text": "Go",
                            "condition": "if codeword ASH is set",
                        }
                    ],
                },
            )
        ]

        result = bar.validate(artifacts)
        assert len(result.warnings) >= 1


class TestNonlinearityBar:
    """Tests for Nonlinearity quality bar."""

    def test_nonlinearity_pass_multiple_paths(self):
        """Test nonlinearity passes for multiple distinct paths."""
        bar = NonlinearityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "start",
                    "start": True,
                    "choices": [
                        {"text": "Left", "target": "left"},
                        {"text": "Right", "target": "right"},
                    ],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={"id": "left", "text": "Left path"},
            ),
            Artifact(
                type="manuscript_section",
                data={"id": "right", "text": "Right path"},
            ),
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_nonlinearity_warn_duplicate_targets(self):
        """Test nonlinearity warns for duplicate choice targets."""
        bar = NonlinearityBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "start",
                    "choices": [
                        {"text": "Option 1", "target": "same"},
                        {"text": "Option 2", "target": "same"},
                    ],
                },
            )
        ]

        result = bar.validate(artifacts)
        assert len(result.warnings) >= 1


class TestDeterminismBar:
    """Tests for Determinism quality bar."""

    def test_determinism_pass_with_params(self):
        """Test determinism passes when params present."""
        bar = DeterminismBar()

        artifacts = [
            Artifact(
                type="visual_asset",
                data={
                    "id": "img1",
                    "params": {
                        "seed": 12345,
                        "model": "stable-diffusion-v2",
                        "prompt_version": "1.0",
                    },
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_determinism_warn_missing_params(self):
        """Test determinism warns when params missing."""
        bar = DeterminismBar()

        artifacts = [
            Artifact(
                type="visual_asset",
                data={
                    "id": "img1",
                },
            )
        ]

        result = bar.validate(artifacts)
        assert len(result.warnings) >= 1


class TestPresentationBar:
    """Tests for Presentation quality bar."""

    def test_presentation_pass_clean_text(self):
        """Test presentation passes for clean player text."""
        bar = PresentationBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "You enter a dark room.",
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_presentation_fail_internal_notes(self):
        """Test presentation fails for internal notes in player text."""
        bar = PresentationBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "You enter. TODO: add more detail",
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is False
        assert len(result.blockers) >= 1

    def test_presentation_fail_spoiler_fields(self):
        """Test presentation fails for spoiler fields in player content."""
        bar = PresentationBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "Content",
                    "author_notes": "Secret plan",
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is False


class TestSpoilerHygieneBar:
    """Tests for Spoiler Hygiene quality bar."""

    def test_spoiler_hygiene_pass_clean_content(self):
        """Test spoiler hygiene passes for clean content."""
        bar = SpoilerHygieneBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "You see a door.",
                },
                metadata={"temperature": "cold", "player_safe": True},
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_spoiler_hygiene_fail_spoiler_field(self):
        """Test spoiler hygiene fails for spoiler fields in player_safe."""
        bar = SpoilerHygieneBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "Content",
                    "spoilers": "Secret",
                },
                metadata={"player_safe": True},
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is False
        assert len(result.blockers) >= 1

    def test_spoiler_hygiene_info_revealing_choice(self):
        """Test spoiler hygiene flags revealing choice text."""
        bar = SpoilerHygieneBar()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "choices": [
                        {
                            "text": "Attack the guard and successfully escape",
                        }
                    ],
                },
            )
        ]

        result = bar.validate(artifacts)
        # Should have info about revealing outcome
        assert len(result.info) >= 1


class TestCanonConflictBar:
    """Test canon conflict quality bar (Layer 6/7)."""

    def test_canon_conflict_pass_no_packages(self):
        """Test canon conflict bar passes with no canon packages"""
        from questfoundry.validation.quality_bars.canon import CanonConflictBar

        bar = CanonConflictBar()
        artifacts = [
            Artifact(type="manuscript_section", data={"id": "section-1"}),
        ]

        result = bar.validate(artifacts)
        assert result.passed is True
        assert len(result.blockers) == 0

    def test_canon_conflict_with_canon_package(self):
        """Test canon conflict bar with canon package"""
        from questfoundry.validation.quality_bars.canon import CanonConflictBar

        bar = CanonConflictBar()
        artifacts = [
            Artifact(
                type="canon_transfer_package",
                artifact_id="CANON-001",
                data={
                    "invariant_canon": [{"facts": ["Dragons sleep for decades"]}],
                    "entity_registry": [],
                },
            )
        ]

        result = bar.validate(artifacts)
        # Should process package without error
        assert result is not None
        assert result.bar_name == "canon_conflict"


class TestTimelineChronologyBar:
    """Test timeline chronology quality bar (Layer 6/7)."""

    def test_timeline_chronology_pass_no_packages(self):
        """Test timeline bar passes with no canon packages"""
        from questfoundry.validation.quality_bars.canon import (
            TimelineChronologyBar,
        )

        bar = TimelineChronologyBar()
        artifacts = [
            Artifact(type="manuscript_section", data={"id": "section-1"}),
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_timeline_chronology_valid(self):
        """Test timeline bar with valid chronology"""
        from questfoundry.validation.quality_bars.canon import (
            TimelineChronologyBar,
        )

        bar = TimelineChronologyBar()
        artifacts = [
            Artifact(
                type="canon_transfer_package",
                artifact_id="CANON-001",
                data={
                    "timeline": {
                        "anchors": [
                            {
                                "anchor_id": "T0",
                                "event": "Foundation",
                                "year": 0,
                                "source": "world-genesis",
                            }
                        ]
                    }
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True


class TestEntityReferenceBar:
    """Test entity reference quality bar (Layer 6/7)."""

    def test_entity_reference_pass_no_packages(self):
        """Test entity bar passes with no canon packages"""
        from questfoundry.validation.quality_bars.canon import (
            EntityReferenceBar,
        )

        bar = EntityReferenceBar()
        artifacts = [
            Artifact(type="manuscript_section", data={"id": "section-1"}),
        ]

        result = bar.validate(artifacts)
        assert result.passed is True

    def test_entity_reference_valid(self):
        """Test entity bar with valid entities"""
        from questfoundry.validation.quality_bars.canon import (
            EntityReferenceBar,
        )

        bar = EntityReferenceBar()
        artifacts = [
            Artifact(
                type="canon_transfer_package",
                artifact_id="CANON-001",
                data={
                    "entity_registry": [
                        {
                            "name": "Dragon Council",
                            "entity_type": "faction",
                            "role": "governing body",
                            "description": "Council of dragons",
                            "source": "world-genesis",
                            "immutable": True,
                        }
                    ]
                },
            )
        ]

        result = bar.validate(artifacts)
        assert result.passed is True


class TestQualityBarRegistry:
    """Test quality bar registry functionality."""

    def test_all_bars_registered(self):
        """Test that all 11 bars are registered."""
        from questfoundry.validation.quality_bars import QUALITY_BARS

        assert len(QUALITY_BARS) == 11
        expected_bars = [
            "integrity",
            "reachability",
            "style",
            "gateways",
            "nonlinearity",
            "determinism",
            "presentation",
            "spoiler_hygiene",
            "canon_conflict",
            "timeline_chronology",
            "entity_reference",
        ]

        for bar_name in expected_bars:
            assert bar_name in QUALITY_BARS

    def test_get_quality_bar(self):
        """Test getting quality bar by name."""
        from questfoundry.validation.quality_bars import get_quality_bar

        IntegrityBarClass = get_quality_bar("integrity")
        assert IntegrityBarClass == IntegrityBar

        with pytest.raises(ValueError):
            get_quality_bar("nonexistent_bar")
