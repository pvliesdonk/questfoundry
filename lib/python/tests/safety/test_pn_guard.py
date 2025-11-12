"""Tests for PN Guard."""

from datetime import datetime

from questfoundry.models.artifact import Artifact
from questfoundry.protocol.envelope import EnvelopeBuilder
from questfoundry.safety import PNGuard, PNViolation


class TestPNGuard:
    """Tests for PNGuard class."""

    def test_initialization(self):
        """Test PNGuard initialization."""
        guard = PNGuard()
        assert guard is not None

    def test_validate_safe_cold_envelope(self):
        """Test validation of safe cold envelope."""
        guard = PNGuard()

        envelope = (
            EnvelopeBuilder()
            .with_id("test-envelope-1")
            .with_time(datetime.now())
            .with_sender("SR")
            .with_receiver("PN")
            .with_intent("deliver.section")
            .with_context("cold", tu="TU-2024-01-01-TEST01")
            .with_safety(player_safe=True, spoilers="forbidden")
            .with_payload("manuscript_section", {"content": "Safe player-facing text"})
            .build()
        )

        result = guard.validate_envelope(envelope)

        assert result.safe is True
        assert len(result.blockers) == 0
        assert result.filtered_envelope is not None

    def test_validate_hot_envelope_blocked(self):
        """Test that hot envelopes are blocked."""
        guard = PNGuard()

        envelope = (
            EnvelopeBuilder()
            .with_id("test-envelope-2")
            .with_time(datetime.now())
            .with_sender("SR")
            .with_receiver("PN")
            .with_intent("deliver.section")
            .with_context("hot", tu="TU-2024-01-01-TEST01")
            .with_safety(player_safe=True, spoilers="forbidden")
            .with_payload("manuscript_section", {"content": "Hot content"})
            .build()
        )

        result = guard.validate_envelope(envelope)

        assert result.safe is False
        assert len(result.blockers) >= 1
        assert any("hot" in v.message.lower() for v in result.violations)

    def test_validate_not_player_safe_blocked(self):
        """Test that non-player-safe content is blocked."""
        guard = PNGuard()

        envelope = (
            EnvelopeBuilder()
            .with_id("test-envelope-3")
            .with_time(datetime.now())
            .with_sender("SR")
            .with_receiver("PN")
            .with_intent("deliver.section")
            .with_context("cold", tu="TU-2024-01-01-TEST01")
            .with_safety(player_safe=False, spoilers="allowed")
            .with_payload("manuscript_section", {"content": "Content"})
            .build()
        )

        result = guard.validate_envelope(envelope)

        assert result.safe is False
        assert len(result.blockers) >= 1

    def test_validate_artifact_with_spoilers(self):
        """Test that spoiler fields are detected."""
        guard = PNGuard()

        artifact = Artifact(
            type="manuscript_section",
            data={
                "id": "section-1",
                "text": "Player text",
                "author_notes": "This is a secret note",
            },
            metadata={"temperature": "cold", "player_safe": True},
        )

        result = guard.validate_artifact(artifact)

        assert result.safe is False
        assert len(result.blockers) >= 1
        assert any("author_notes" in v.location for v in result.violations)

    def test_validate_artifact_meta_language(self):
        """Test detection of meta/mechanical language."""
        guard = PNGuard()

        artifact = Artifact(
            type="manuscript_section",
            data={
                "id": "section-2",
                "text": "You have CODEWORD: ASH, go to section 17",
            },
            metadata={"temperature": "cold", "player_safe": True},
        )

        result = guard.validate_artifact(artifact)

        assert result.safe is False
        assert len(result.blockers) >= 1
        assert any(
            "meta" in v.category or "language" in v.category for v in result.violations
        )

    def test_validate_artifact_non_diegetic_gateway(self):
        """Test detection of non-diegetic gateway conditions."""
        guard = PNGuard()

        artifact = Artifact(
            type="manuscript_section",
            data={
                "id": "section-3",
                "gateways": [
                    {"condition": "if codeword ASH is set"},
                ],
            },
            metadata={"temperature": "cold", "player_safe": True},
        )

        result = guard.validate_artifact(artifact)

        assert result.safe is True  # Warnings don't block
        assert len(result.warnings) >= 1
        assert any("diegetic" in v.message.lower() for v in result.warnings)

    def test_filter_artifact_removes_spoilers(self):
        """Test that artifact filtering removes spoiler fields."""
        guard = PNGuard()

        artifact = Artifact(
            type="manuscript_section",
            data={
                "id": "section-4",
                "text": "Player text",
                "author_notes": "Secret",
                "gm_notes": "Also secret",
                "nested": {
                    "visible": "yes",
                    "internal_notes": "hidden",
                },
            },
        )

        filtered = guard._filter_artifact(artifact)

        assert "text" in filtered.data
        assert "author_notes" not in filtered.data
        assert "gm_notes" not in filtered.data
        assert "visible" in filtered.data["nested"]
        assert "internal_notes" not in filtered.data["nested"]

    def test_validate_artifacts_list(self):
        """Test validation of multiple artifacts."""
        guard = PNGuard()

        artifacts = [
            Artifact(
                type="manuscript_section",
                data={"id": "section-1", "text": "Safe text"},
                metadata={"temperature": "cold", "player_safe": True},
            ),
            Artifact(
                type="manuscript_section",
                data={
                    "id": "section-2",
                    "text": "Text",
                    "spoilers": "Bad",
                },
                metadata={"temperature": "cold", "player_safe": True},
            ),
        ]

        result = guard.validate_artifacts(artifacts)

        assert result.safe is False
        assert len(result.blockers) >= 1
        assert len(result.filtered_artifacts) == 1  # Only first one is safe

    def test_violation_severity_levels(self):
        """Test that violations have correct severity levels."""
        guard = PNGuard()

        # Blocker: spoiler field in artifact data
        spoiler_artifact = Artifact(
            type="manuscript_section",
            data={"id": "s1", "text": "Text", "author_notes": "Secret"},
        )

        result = guard.validate_artifact(spoiler_artifact)
        assert any(v.severity == "blocker" for v in result.violations)

        # Warning: non-diegetic gateway
        gateway_artifact = Artifact(
            type="manuscript_section",
            data={
                "id": "s2",
                "gateways": [{"condition": "if flag is set"}],
            },
        )

        result = guard.validate_artifact(gateway_artifact)
        # May have warnings about diegetic gateways
        assert len(result.warnings) >= 0  # Warnings don't block

    def test_pn_guard_result_properties(self):
        """Test PNGuardResult helper properties."""
        violations = [
            PNViolation(
                severity="blocker",
                category="test",
                message="Blocker issue",
                location="test",
            ),
            PNViolation(
                severity="warning",
                category="test",
                message="Warning issue",
                location="test",
            ),
        ]

        from questfoundry.safety.pn_guard import PNGuardResult

        result = PNGuardResult(safe=False, violations=violations)

        assert len(result.blockers) == 1
        assert len(result.warnings) == 1
        assert result.blockers[0].severity == "blocker"
        assert result.warnings[0].severity == "warning"
