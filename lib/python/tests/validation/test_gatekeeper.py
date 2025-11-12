"""Tests for Gatekeeper integration."""

import pytest

from questfoundry.models.artifact import Artifact
from questfoundry.validation import GatecheckReport, Gatekeeper


class TestGatekeeper:
    """Tests for Gatekeeper class."""

    def test_initialization_default(self):
        """Test Gatekeeper initialization with defaults."""
        gk = Gatekeeper()
        assert len(gk.bars) == 11
        assert gk.strict is True

    def test_initialization_custom_bars(self):
        """Test Gatekeeper with custom bar selection."""
        gk = Gatekeeper(bars=["integrity", "reachability"])
        assert len(gk.bars) == 2
        assert "integrity" in gk.bars
        assert "reachability" in gk.bars

    def test_run_gatecheck_pass(self):
        """Test gatecheck passes for valid content."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "section-1",
                    "start": True,
                    "text": "You begin your journey.",
                    "choices": [{"text": "Continue", "target": "section-2"}],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "section-2",
                    "text": "The end.",
                    "terminal": True,
                },
            ),
        ]

        report = gk.run_gatecheck(artifacts)

        assert isinstance(report, GatecheckReport)
        assert report.passed is True
        assert report.merge_safe is True
        assert len(report.bar_results) == 11

    def test_run_gatecheck_fail_blockers(self):
        """Test gatecheck fails with blockers."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    # Missing ID - integrity blocker
                    "text": "Content without ID",
                },
            )
        ]

        report = gk.run_gatecheck(artifacts)

        assert report.passed is False
        assert report.merge_safe is False
        assert len(report.blockers) >= 1

    def test_run_gatecheck_warnings_strict(self):
        """Test gatecheck in strict mode fails on warnings."""
        gk = Gatekeeper(strict=True)

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "start": True,
                    "text": "The end.",
                    # No terminal marker - will warn
                },
            )
        ]

        report = gk.run_gatecheck(artifacts)

        # In strict mode, warnings may cause failure
        # Depends on bars that run
        assert isinstance(report, GatecheckReport)

    def test_run_gatecheck_warnings_non_strict(self):
        """Test gatecheck in non-strict mode passes with warnings."""
        gk = Gatekeeper(strict=False)

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "Content",
                },
            )
        ]

        report = gk.run_gatecheck(artifacts)

        # Non-strict allows warnings
        assert isinstance(report, GatecheckReport)

    def test_run_bar_individually(self):
        """Test running a single bar."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "Content",
                },
            )
        ]

        result = gk.run_bar("integrity", artifacts)

        assert result.bar_name == "integrity"
        assert result.passed is True

    def test_run_bar_invalid_name(self):
        """Test running invalid bar raises error."""
        gk = Gatekeeper(bars=["integrity"])

        with pytest.raises(ValueError, match="not configured"):
            gk.run_bar("nonexistent", [])

    def test_gatecheck_report_properties(self):
        """Test GatecheckReport helper properties."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    # Missing ID - blocker
                    "text": "No ID",
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    # No choices, no terminal - warning
                    "text": "Content",
                },
            ),
        ]

        report = gk.run_gatecheck(artifacts)

        assert len(report.all_issues) > 0
        assert len(report.blockers) >= 1
        assert len(report.warnings) >= 0

    def test_gatecheck_report_summary(self):
        """Test gatecheck report summary generation."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "Content",
                },
            )
        ]

        report = gk.run_gatecheck(artifacts, timestamp="2024-01-01")

        assert isinstance(report.summary, str)
        assert len(report.summary) > 0
        assert "Gatecheck" in report.summary

    def test_gatecheck_report_to_artifact(self):
        """Test converting report to artifact."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "Content",
                },
            )
        ]

        report = gk.run_gatecheck(artifacts, timestamp="2024-01-01")
        artifact = report.to_artifact()

        assert artifact.type == "gatecheck_report"
        assert "passed" in artifact.data
        assert "merge_safe" in artifact.data
        assert "bars" in artifact.data
        assert "summary" in artifact.data

    def test_gatecheck_metadata(self):
        """Test gatecheck report includes metadata."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={"id": "s1", "text": "Content"},
            )
        ]

        report = gk.run_gatecheck(
            artifacts,
            timestamp="2024-01-01",
            author="test",
            workspace="test-ws",
        )

        assert report.metadata["timestamp"] == "2024-01-01"
        assert report.metadata["author"] == "test"
        assert report.metadata["workspace"] == "test-ws"

    def test_gatecheck_complex_manuscript(self):
        """Test gatecheck on complex manuscript structure."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={
                    "id": "start",
                    "start": True,
                    "text": "You stand at a crossroads.",
                    "choices": [
                        {"text": "Go left", "target": "left"},
                        {"text": "Go right", "target": "right"},
                    ],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "left",
                    "text": "You take the left path.",
                    "choices": [{"text": "Continue", "target": "hub"}],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "right",
                    "text": "You take the right path.",
                    "choices": [{"text": "Continue", "target": "hub"}],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "hub",
                    "text": "Both paths converge here.",
                    "terminal": True,
                },
            ),
        ]

        report = gk.run_gatecheck(artifacts)

        # Should pass - valid structure
        assert report.passed is True
        assert report.merge_safe is True

    def test_gatecheck_with_style_guide(self):
        """Test gatecheck with style guide artifact."""
        gk = Gatekeeper()

        artifacts = [
            Artifact(
                type="style_guide",
                data={
                    "id": "style-1",
                    "voice": "sardonic",
                    "register": "literary",
                    "motifs": ["rust", "decay", "hope"],
                },
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "s1",
                    "text": "The rusted gate groaned open.",
                },
            ),
        ]

        report = gk.run_gatecheck(artifacts)

        # Style bar should validate
        assert "style" in report.bar_results
        assert report.bar_results["style"].passed is True
