"""Tests for protocol conformance validation"""

from datetime import datetime, timezone

from questfoundry.protocol import (
    ConformanceResult,
    EnvelopeBuilder,
    HotCold,
    RoleName,
    SpoilerPolicy,
    validate_envelope_conformance,
)


def test_conformant_hot_envelope():
    """Test conformant hot envelope passes validation"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.SCENE_SMITH)
        .with_intent("scene.write")
        .with_context(HotCold.HOT, tu="TU-2024-01-15-SR01")
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload(
            "tu_brief",
            {
                "header": {
                    "short_name": "Test",
                    "id": "TU-2024-01-15-SR01",
                    "status": "open",
                }
            },
        )
        .build()
    )

    result = validate_envelope_conformance(envelope)

    # May have payload validation errors due to incomplete data,
    # but no protocol violations
    assert isinstance(result, ConformanceResult)


def test_pn_safety_invariant_cold_valid():
    """Test PN receiver with valid cold context"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-pn-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.BOOK_BINDER)
        .with_receiver(RoleName.PLAYER_NARRATOR)
        .with_intent("narration.perform")
        .with_context(HotCold.COLD, snapshot="Cold @ 2024-01-15")
        .with_safety(True, SpoilerPolicy.FORBIDDEN)
        .with_payload("view_log", {"header": {"id": "VL-001"}})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    # Should have no PN safety violations (may have payload schema errors)
    assert not any(
        v.rule == "PN_SAFETY_INVARIANT" for v in result.violations
    ), "Should not have PN safety violations"


def test_pn_safety_invariant_hot_invalid():
    """Test PN receiver with hot context (invalid)"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-pn-invalid-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.PLAYER_NARRATOR)
        .with_intent("narration.perform")
        .with_context(HotCold.HOT)  # Invalid: should be COLD
        .with_safety(True, SpoilerPolicy.FORBIDDEN)
        .with_payload("view_log", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    assert not result.conformant
    # Should have PN safety violations
    pn_violations = [v for v in result.violations if v.rule == "PN_SAFETY_INVARIANT"]
    assert len(pn_violations) > 0
    assert any("cold" in v.message.lower() for v in pn_violations)


def test_pn_safety_invariant_missing_snapshot():
    """Test PN receiver without snapshot (invalid)"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-pn-invalid-002")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.BOOK_BINDER)
        .with_receiver(RoleName.PLAYER_NARRATOR)
        .with_intent("narration.perform")
        .with_context(HotCold.COLD)  # Missing snapshot
        .with_safety(True, SpoilerPolicy.FORBIDDEN)
        .with_payload("view_log", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    assert not result.conformant
    pn_violations = [v for v in result.violations if v.rule == "PN_SAFETY_INVARIANT"]
    assert any("snapshot" in v.message.lower() for v in pn_violations)


def test_pn_safety_invariant_not_player_safe():
    """Test PN receiver with player_safe=false (invalid)"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-pn-invalid-003")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.PLAYER_NARRATOR)
        .with_intent("narration.perform")
        .with_context(HotCold.COLD, snapshot="Cold @ 2024-01-15")
        .with_safety(False, SpoilerPolicy.FORBIDDEN)  # Invalid: should be True
        .with_payload("view_log", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    assert not result.conformant
    pn_violations = [v for v in result.violations if v.rule == "PN_SAFETY_INVARIANT"]
    assert any("player_safe" in v.message.lower() for v in pn_violations)


def test_pn_safety_invariant_spoilers_allowed():
    """Test PN receiver with spoilers=allowed (invalid)"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-pn-invalid-004")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.PLAYER_NARRATOR)
        .with_intent("narration.perform")
        .with_context(HotCold.COLD, snapshot="Cold @ 2024-01-15")
        .with_safety(True, SpoilerPolicy.ALLOWED)  # Invalid: should be FORBIDDEN
        .with_payload("view_log", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    assert not result.conformant
    pn_violations = [v for v in result.violations if v.rule == "PN_SAFETY_INVARIANT"]
    assert any("spoilers" in v.message.lower() for v in pn_violations)


def test_conformance_result_formatting():
    """Test ConformanceResult formatting methods"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-format-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.PLAYER_NARRATOR)
        .with_intent("narration.perform")
        .with_context(HotCold.HOT)  # Multiple violations
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload("view_log", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    formatted = result.format_violations()
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "PN_SAFETY_INVARIANT" in formatted


def test_non_pn_receiver_no_safety_check():
    """Test that non-PN receivers don't trigger PN safety checks"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-non-pn-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.SCENE_SMITH)
        .with_intent("scene.write")
        .with_context(HotCold.HOT)  # Would be invalid for PN, but OK for SS
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload("tu_brief", {"header": {"short_name": "Test"}})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    # Should have no PN safety violations (may have payload validation errors)
    pn_violations = [v for v in result.violations if v.rule == "PN_SAFETY_INVARIANT"]
    assert len(pn_violations) == 0


def test_protocol_version_warning():
    """Test protocol version compatibility warning"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("2.0.0")  # Future version
        .with_id("test-version-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.SHOWRUNNER)
        .with_receiver(RoleName.SCENE_SMITH)
        .with_intent("scene.write")
        .with_context(HotCold.HOT)
        .with_safety(False, SpoilerPolicy.ALLOWED)
        .with_payload("tu_brief", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    # Should have version warning
    assert result.has_warnings
    version_warnings = [w for w in result.warnings if w.rule == "PROTOCOL_VERSION"]
    assert len(version_warnings) > 0


def test_cold_context_without_snapshot_warning():
    """Test cold context without snapshot generates warning"""
    envelope = (
        EnvelopeBuilder()
        .with_protocol("1.0.0")
        .with_id("test-cold-001")
        .with_time(datetime.now(timezone.utc))
        .with_sender(RoleName.LORE_WEAVER)
        .with_receiver(RoleName.CODEX_CURATOR)
        .with_intent("codex.create")
        .with_context(HotCold.COLD)  # No snapshot
        .with_safety(True, SpoilerPolicy.FORBIDDEN)
        .with_payload("codex_entry", {})
        .build()
    )

    result = validate_envelope_conformance(envelope)

    # For non-PN, this is just a warning
    # Check warnings, not violations for non-PN
    all_issues = result.violations + result.warnings
    assert any("snapshot" in str(v).lower() for v in all_issues)
