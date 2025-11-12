"""
PN Guard: Enforces Player-Narrator boundaries.

The PN Guard ensures that only safe, cold, player-appropriate content
reaches the Player-Narrator (PN) surface. It filters envelopes, strips
spoilers, validates diegetic gateways, and prevents meta/mechanical
language from appearing in player-facing text.

Key responsibilities:
- Filter to cold + player_safe content only
- Strip spoiler fields from artifacts
- Validate gateways use diegetic (in-world) conditions
- Detect and block codewords/mechanics in player text
- Generate clear violation reports
"""

import re
from dataclasses import dataclass, field
from typing import Any

from ..models.artifact import Artifact
from ..protocol.envelope import Envelope, Payload


@dataclass
class PNViolation:
    """
    A violation of PN boundaries.

    Attributes:
        severity: 'blocker' (must fix) or 'warning' (should review)
        category: Type of violation (spoiler, hot_content, meta_language, etc.)
        message: Human-readable description
        location: Where the violation was found (artifact_id, field, etc.)
        suggestion: How to fix it
    """

    severity: str  # 'blocker' or 'warning'
    category: str
    message: str
    location: str
    suggestion: str = ""


@dataclass
class PNGuardResult:
    """
    Result of PN Guard validation.

    Attributes:
        safe: Whether content passes PN boundaries
        violations: List of violations found
        filtered_envelope: Envelope with spoilers stripped (if safe)
        filtered_artifacts: Artifacts with spoilers stripped (if safe)
    """

    safe: bool
    violations: list[PNViolation] = field(default_factory=list)
    filtered_envelope: Envelope | None = None
    filtered_artifacts: list[Artifact] = field(default_factory=list)

    @property
    def blockers(self) -> list[PNViolation]:
        """Get only blocker violations."""
        return [v for v in self.violations if v.severity == "blocker"]

    @property
    def warnings(self) -> list[PNViolation]:
        """Get only warning violations."""
        return [v for v in self.violations if v.severity == "warning"]


class PNGuard:
    """
    PN Guard: Enforces Player-Narrator boundaries.

    Validates and filters content to ensure only safe, cold, player-appropriate
    material reaches the PN surface.
    """

    # Forbidden meta/mechanical patterns in player text
    META_PATTERNS = [
        r"\bcodeword\b",
        r"\bCODEWORD\b",
        r"\bstate\s*=",
        r"\bset\s+flag\b",
        r"section\s+\d+",
        r"\bgo\s+to\s+section\b",
        r"\bRNG\b",
        r"\bseed\b.*\d+",
        r"\bmodel\b.*\bversion\b",
        r"\bDAW\b",
        r"\bprompt\s+version\b",
    ]

    # Spoiler fields that must be stripped
    SPOILER_FIELDS = [
        "author_notes",
        "internal_notes",
        "gm_notes",
        "spoilers",
        "hidden",
        "secret",
        "canon_notes",
    ]

    def __init__(self) -> None:
        """Initialize PN Guard with meta pattern regexes."""
        self._meta_regex = re.compile("|".join(self.META_PATTERNS), re.IGNORECASE)

    def validate_envelope(self, envelope: Envelope) -> PNGuardResult:
        """
        Validate an envelope for PN safety.

        Checks:
        1. Only cold content (no hot)
        2. Only player_safe content
        3. No spoiler leaks
        4. No meta/mechanical language

        Args:
            envelope: Envelope to validate

        Returns:
            PNGuardResult with violations and filtered content
        """
        violations: list[PNViolation] = []

        # Check envelope context and safety
        hot_cold = envelope.context.hot_cold
        player_safe = envelope.safety.player_safe

        # Validate temperature (hot/cold)
        if hot_cold == "hot":
            violations.append(
                PNViolation(
                    severity="blocker",
                    category="hot_content",
                    message="Hot content cannot be sent to PN",
                    location=f"envelope:{envelope.id}",
                    suggestion=(
                        "Only cold (merged to Cold SoT) content can be exposed to PN"
                    ),
                )
            )

        # Validate player_safe flag
        if not player_safe:
            violations.append(
                PNViolation(
                    severity="blocker",
                    category="not_player_safe",
                    message="Content not marked as player_safe",
                    location=f"envelope:{envelope.id}",
                    suggestion=(
                        "Mark content as player_safe after removing "
                        "spoilers and internal notes"
                    ),
                )
            )

        # Validate payload if present
        if envelope.payload:
            payload_violations = self._validate_payload(
                envelope.payload.data, f"envelope:{envelope.id}"
            )
            violations.extend(payload_violations)

        # Create filtered envelope if safe
        filtered_envelope = None
        if not any(v.severity == "blocker" for v in violations):
            filtered_envelope = self._filter_envelope(envelope)

        return PNGuardResult(
            safe=not any(v.severity == "blocker" for v in violations),
            violations=violations,
            filtered_envelope=filtered_envelope,
        )

    def validate_artifacts(self, artifacts: list[Artifact]) -> PNGuardResult:
        """
        Validate a list of artifacts for PN safety.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            PNGuardResult with violations and filtered artifacts
        """
        violations: list[PNViolation] = []
        filtered_artifacts: list[Artifact] = []

        for artifact in artifacts:
            result = self.validate_artifact(artifact)
            violations.extend(result.violations)
            if result.safe and result.filtered_artifacts:
                filtered_artifacts.extend(result.filtered_artifacts)

        return PNGuardResult(
            safe=not any(v.severity == "blocker" for v in violations),
            violations=violations,
            filtered_artifacts=filtered_artifacts,
        )

    def validate_artifact(self, artifact: Artifact) -> PNGuardResult:
        """
        Validate a single artifact for PN safety.

        Args:
            artifact: Artifact to validate

        Returns:
            PNGuardResult with violations and filtered artifact
        """
        violations: list[PNViolation] = []

        artifact_id = artifact.data.get("id", "unknown")

        # Check data for violations
        data_violations = self._validate_payload(
            artifact.data, f"artifact:{artifact_id}"
        )
        violations.extend(data_violations)

        # Create filtered artifact if safe
        filtered_artifact = None
        if not any(v.severity == "blocker" for v in violations):
            filtered_artifact = self._filter_artifact(artifact)

        return PNGuardResult(
            safe=not any(v.severity == "blocker" for v in violations),
            violations=violations,
            filtered_artifacts=[filtered_artifact] if filtered_artifact else [],
        )

    def _validate_payload(
        self, payload: dict[str, Any], location: str
    ) -> list[PNViolation]:
        """
        Validate payload data for PN violations.

        Args:
            payload: Data to validate
            location: Where this payload is from (for violation reporting)

        Returns:
            List of violations found
        """
        violations: list[PNViolation] = []

        # Check for spoiler fields
        for field_name in self.SPOILER_FIELDS:
            if field_name in payload and payload[field_name]:
                violations.append(
                    PNViolation(
                        severity="blocker",
                        category="spoiler_field",
                        message=(
                            f"Spoiler field '{field_name}' present in "
                            f"player-facing content"
                        ),
                        location=f"{location}.{field_name}",
                        suggestion=(
                            f"Remove the '{field_name}' field or move to canon notes"
                        ),
                    )
                )

        # Check text fields for meta language
        violations.extend(self._check_meta_language(payload, location))

        # Check gateways for diegetic violations
        if "gateways" in payload or "conditions" in payload:
            violations.extend(self._check_gateway_diegetic(payload, location))

        return violations

    def _check_meta_language(
        self, data: dict[str, Any], location: str
    ) -> list[PNViolation]:
        """
        Check for meta/mechanical language in text fields.

        Args:
            data: Data to check
            location: Location for violation reporting

        Returns:
            List of violations found
        """
        violations: list[PNViolation] = []

        # Fields to check for player-facing text
        text_fields = ["text", "body", "content", "description", "choice"]

        for field_name in text_fields:
            if field_name in data and isinstance(data[field_name], str):
                matches = self._meta_regex.findall(data[field_name])
                if matches:
                    violations.append(
                        PNViolation(
                            severity="blocker",
                            category="meta_language",
                            message=(
                                f"Meta/mechanical language found in "
                                f"'{field_name}': {matches[:3]}"
                            ),
                            location=f"{location}.{field_name}",
                            suggestion=(
                                "Use diegetic (in-world) phrasing "
                                "instead of technical terms"
                            ),
                        )
                    )

        # Recursively check nested dicts and lists
        for key, value in data.items():
            if isinstance(value, dict):
                violations.extend(self._check_meta_language(value, f"{location}.{key}"))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(
                            self._check_meta_language(item, f"{location}.{key}[{i}]")
                        )

        return violations

    def _check_gateway_diegetic(
        self, data: dict[str, Any], location: str
    ) -> list[PNViolation]:
        """
        Check that gateway conditions are diegetic (in-world).

        Args:
            data: Data containing gateways
            location: Location for violation reporting

        Returns:
            List of violations found
        """
        violations: list[PNViolation] = []

        # Check gateways
        gateways = data.get("gateways", data.get("conditions", []))
        if not isinstance(gateways, list):
            gateways = [gateways]

        for i, gateway in enumerate(gateways):
            if not isinstance(gateway, dict):
                continue

            # Check condition field
            condition = gateway.get("condition", "")
            if isinstance(condition, str):
                # Look for non-diegetic patterns
                if re.search(
                    r"(codeword|flag|state|variable)", condition, re.IGNORECASE
                ):
                    violations.append(
                        PNViolation(
                            severity="warning",
                            category="non_diegetic_gateway",
                            message=(
                                f"Gateway condition may not be diegetic: "
                                f"'{condition[:50]}'"
                            ),
                            location=f"{location}.gateways[{i}]",
                            suggestion=(
                                "Use in-world items, knowledge, or reputation "
                                "instead of technical state"
                            ),
                        )
                    )

        return violations

    def _filter_envelope(self, envelope: Envelope) -> Envelope:
        """
        Create a filtered copy of envelope with spoilers stripped.

        Args:
            envelope: Envelope to filter

        Returns:
            Filtered envelope safe for PN
        """
        # Create filtered payload data
        filtered_data = self._filter_data(envelope.payload.data)

        # Create new payload with filtered data
        filtered_payload = Payload(type=envelope.payload.type, data=filtered_data)

        # Return new envelope with filtered payload
        return Envelope(
            protocol=envelope.protocol,
            id=envelope.id,
            time=envelope.time,
            sender=envelope.sender,
            receiver=envelope.receiver,
            intent=envelope.intent,
            correlation_id=envelope.correlation_id,
            reply_to=envelope.reply_to,
            context=envelope.context,
            safety=envelope.safety,
            payload=filtered_payload,
            refs=envelope.refs,
        )

    def _filter_artifact(self, artifact: Artifact) -> Artifact:
        """
        Create a filtered copy of artifact with spoilers stripped.

        Args:
            artifact: Artifact to filter

        Returns:
            Filtered artifact safe for PN
        """
        filtered_data = self._filter_data(artifact.data)

        return Artifact(
            type=artifact.type,
            data=filtered_data,
            metadata=artifact.metadata,
        )

    def _filter_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively filter data to remove spoiler fields.

        Args:
            data: Data to filter

        Returns:
            Filtered data without spoiler fields
        """
        if not isinstance(data, dict):
            return data

        filtered: dict[str, Any] = {}

        for key, value in data.items():
            # Skip spoiler fields
            if key in self.SPOILER_FIELDS:
                continue

            # Recursively filter nested data
            if isinstance(value, dict):
                filtered[key] = self._filter_data(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value

        return filtered
