"""Protocol conformance validation"""

import logging
from dataclasses import dataclass, field

from ..validators import validate_artifact
from .envelope import Envelope
from .types import HotCold, RoleName, SpoilerPolicy

logger = logging.getLogger(__name__)


@dataclass
class ConformanceViolation:
    """Represents a protocol conformance violation"""

    rule: str
    message: str
    severity: str = "error"  # error, warning
    reference: str | None = None


@dataclass
class ConformanceResult:
    """Result of protocol conformance validation"""

    conformant: bool
    violations: list[ConformanceViolation] = field(default_factory=list)
    warnings: list[ConformanceViolation] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level violations"""
        return any(v.severity == "error" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0

    def format_violations(self) -> str:
        """Format all violations as a multi-line string"""
        if not self.violations:
            return "No violations"
        lines = []
        for v in self.violations:
            line = f"[{v.severity.upper()}] {v.rule}: {v.message}"
            if v.reference:
                line += f" (ref: {v.reference})"
            lines.append(line)
        return "\n".join(lines)

    def format_warnings(self) -> str:
        """Format all warnings as a multi-line string"""
        if not self.warnings:
            return "No warnings"
        return "\n".join(f"[WARNING] {w.rule}: {w.message}" for w in self.warnings)


def validate_envelope_conformance(envelope: Envelope) -> ConformanceResult:
    """
    Validate envelope conformance against Layer 4 protocol rules.

    Checks:
    - PN safety invariant (hot/cold, player_safe flags)
    - Protocol version compatibility
    - Context field requirements
    - Payload schema validation

    Args:
        envelope: Envelope to validate

    Returns:
        ConformanceResult with violations and warnings
    """
    logger.trace("Starting conformance validation for envelope %s", envelope.id)
    violations: list[ConformanceViolation] = []
    warnings: list[ConformanceViolation] = []

    # Check PN Safety Invariant
    if envelope.receiver.role == RoleName.PLAYER_NARRATOR:
        logger.debug("Checking PN safety invariant for envelope %s to PN", envelope.id)
        pn_violations = _check_pn_safety_invariant(envelope)
        violations.extend(pn_violations)
        if pn_violations:
            logger.warning(
                "PN safety invariant violations found: %d", len(pn_violations)
            )

    # Check context requirements
    logger.trace("Checking context requirements for envelope %s", envelope.id)
    context_warnings = _check_context_requirements(envelope)
    warnings.extend(context_warnings)

    # Check payload schema validation
    logger.trace("Validating payload schema for type %s", envelope.payload.type)
    payload_result = validate_artifact(envelope.payload.data, envelope.payload.type)
    if not payload_result.valid:
        logger.warning(
            "Payload schema validation failed for type %s: %d errors",
            envelope.payload.type,
            len(payload_result.errors),
        )
        violations.append(
            ConformanceViolation(
                rule="PAYLOAD_SCHEMA_VALIDATION",
                message=(
                    f"Payload data does not conform to {envelope.payload.type} schema: "
                    f"{len(payload_result.errors)} errors"
                ),
                severity="error",
            )
        )

    # Check protocol version compatibility
    logger.trace(
        "Checking protocol version %s compatibility", envelope.protocol.version
    )
    version_warnings = _check_protocol_version(envelope)
    warnings.extend(version_warnings)

    result = ConformanceResult(
        conformant=len(violations) == 0, violations=violations, warnings=warnings
    )

    if result.conformant:
        logger.debug("Envelope %s passed conformance validation", envelope.id)
    else:
        logger.warning(
            "Envelope %s failed conformance validation with %d violations",
            envelope.id,
            len(violations),
        )

    return result


def _check_pn_safety_invariant(envelope: Envelope) -> list[ConformanceViolation]:
    """
    Validate PN Safety Invariant.

    When receiver.role = "PN", MUST satisfy ALL:
    - context.hot_cold = "cold"
    - context.snapshot MUST be present
    - safety.player_safe = true
    - safety.spoilers = "forbidden"
    """
    violations: list[ConformanceViolation] = []

    if envelope.context.hot_cold != HotCold.COLD:
        violations.append(
            ConformanceViolation(
                rule="PN_SAFETY_INVARIANT",
                message="PN receiver requires context.hot_cold = 'cold'",
                severity="error",
                reference="00-north-star/PN_PRINCIPLES.md",
            )
        )

    if not envelope.context.snapshot:
        violations.append(
            ConformanceViolation(
                rule="PN_SAFETY_INVARIANT",
                message="PN receiver requires context.snapshot to be present",
                severity="error",
                reference="00-north-star/PN_PRINCIPLES.md",
            )
        )

    if not envelope.safety.player_safe:
        violations.append(
            ConformanceViolation(
                rule="PN_SAFETY_INVARIANT",
                message="PN receiver requires safety.player_safe = true",
                severity="error",
                reference="00-north-star/PN_PRINCIPLES.md",
            )
        )

    if envelope.safety.spoilers != SpoilerPolicy.FORBIDDEN:
        violations.append(
            ConformanceViolation(
                rule="PN_SAFETY_INVARIANT",
                message="PN receiver requires safety.spoilers = 'forbidden'",
                severity="error",
                reference="00-north-star/PN_PRINCIPLES.md",
            )
        )

    return violations


def _check_context_requirements(envelope: Envelope) -> list[ConformanceViolation]:
    """Check context field requirements (warnings only)"""
    warnings: list[ConformanceViolation] = []

    # Cold context should have snapshot (warning for non-PN roles only)
    # PN roles already have this as an error-level check in the safety invariant
    if (
        envelope.receiver.role != RoleName.PLAYER_NARRATOR
        and envelope.context.hot_cold == HotCold.COLD
        and not envelope.context.snapshot
    ):
        warnings.append(
            ConformanceViolation(
                rule="CONTEXT_SNAPSHOT_REQUIRED",
                message="Cold context should include snapshot reference",
                severity="warning",
            )
        )

    return warnings


def _check_protocol_version(envelope: Envelope) -> list[ConformanceViolation]:
    """
    Check protocol version compatibility.

    Note: Version format is already validated by Pydantic semver pattern,
    so we only need to check major version compatibility.
    """
    warnings: list[ConformanceViolation] = []

    # Parse version - format is guaranteed valid by Pydantic
    version_parts = envelope.protocol.version.split(".")
    major = int(version_parts[0])

    # For now, we only support version 1.x.x
    if major != 1:
        warnings.append(
            ConformanceViolation(
                rule="PROTOCOL_VERSION",
                message=(
                    f"Protocol major version {major} may not be compatible "
                    "(expecting 1.x.x)"
                ),
                severity="warning",
            )
        )

    return warnings
