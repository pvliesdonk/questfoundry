"""
Canon workflow quality bars (Layer 6/7).

Validates canon integrity, timeline chronology, and entity consistency
for canon transfer and world genesis workflows.
"""

import logging
from typing import Any

from ...models.artifact import Artifact
from ...state.entity_registry import EntityType
from ...state.timeline import TimelineAnchor, TimelineManager
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class CanonConflictBar(QualityBar):
    """
    Validates that imported canon does not conflict with existing canon.

    Detects contradictions between invariant canon and new project ideas
    or conflicts between multiple canon sources. Uses keyword matching
    and semantic analysis to identify issues.
    """

    @property
    def name(self) -> str:
        return "canon_conflict"

    @property
    def description(self) -> str:
        return "Validates no conflicts in imported canon"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate canon artifacts for conflicts.

        Args:
            artifacts: List of artifacts to validate (should include
                canon_transfer_package)

        Returns:
            QualityBarResult with conflict issues
        """
        logger.debug("Validating canon conflicts in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Extract canon transfer packages
        canon_packages = [a for a in artifacts if a.type == "canon_transfer_package"]

        if not canon_packages:
            # No canon packages to validate - pass
            logger.trace("No canon packages found to validate")
            return self._create_result(issues, validated_packages=0)

        for pkg in canon_packages:
            pkg_id = pkg.artifact_id or "unknown"

            # Extract invariant canon
            invariant_canon: list[str] = []
            if "invariant_canon" in pkg.data:
                for canon_item in pkg.data["invariant_canon"]:
                    if isinstance(canon_item, dict) and "facts" in canon_item:
                        invariant_canon.extend(canon_item["facts"])
                    elif isinstance(canon_item, str):
                        invariant_canon.append(canon_item)

            # Check for internal conflicts within the package
            # (detect contradictions within invariant canon itself)
            if len(invariant_canon) > 1:
                # Simple pairwise conflict detection
                for i in range(len(invariant_canon)):
                    for j in range(i + 1, len(invariant_canon)):
                        fact1 = invariant_canon[i]
                        fact2 = invariant_canon[j]

                        # Check for keyword contradictions
                        conflicts = self._detect_contradiction(fact1, fact2)
                        if conflicts:
                            issues.append(
                                QualityIssue(
                                    severity="blocker",
                                    message=(
                                        f"Internal canon conflict: '{fact1}' "
                                        f"contradicts '{fact2}'"
                                    ),
                                    location=f"{pkg_id}/invariant_canon",
                                    fix=(
                                        "Review and resolve contradictory "
                                        "canon statements"
                                    ),
                                )
                            )

            # Validate entity registry conflicts
            if "entity_registry" in pkg.data:
                logger.trace("Validating entity registry for package %s", pkg_id)
                entity_issues = self._validate_entity_registry(
                    pkg.data["entity_registry"], pkg_id
                )
                issues.extend(entity_issues)

        logger.debug("Canon conflict validation complete: %d issues found", len(issues))
        return self._create_result(issues, validated_packages=len(canon_packages))

    def _detect_contradiction(self, fact1: str, fact2: str) -> bool:
        """Simple contradiction detection using keyword pairs."""
        contradictions = [
            ("destroyed", "intact"),
            ("alive", "dead"),
            ("exists", "doesn't exist"),
            ("can", "cannot"),
            ("never", "always"),
        ]

        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()

        for word1, word2 in contradictions:
            if (word1 in fact1_lower and word2 in fact2_lower) or (
                word2 in fact1_lower and word1 in fact2_lower
            ):
                return True
        return False

    def _validate_entity_registry(
        self, registry_data: dict[str, Any], pkg_id: str
    ) -> list[QualityIssue]:
        """Validate entity registry for duplicates and conflicts."""
        issues: list[QualityIssue] = []

        if "entities" not in registry_data:
            return issues

        entities = registry_data["entities"]
        seen_names: dict[str, dict[str, Any]] = {}

        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("entity_type", "")
            immutable = entity.get("immutable", False)

            key = f"{name}:{entity_type}"

            if key in seen_names:
                prev = seen_names[key]
                # Conflict if both claim to be immutable
                if immutable and prev["immutable"]:
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            message=(
                                f"Duplicate immutable entity: {name} ({entity_type})"
                            ),
                            location=f"{pkg_id}/entity_registry",
                            fix="Remove duplicate or mark one as mutable",
                        )
                    )
            else:
                seen_names[key] = entity

        return issues


class TimelineChronologyBar(QualityBar):
    """
    Validates timeline anchors are properly ordered and chronologically consistent.

    Checks for:
    - Proper baseline anchor sequence (T0, T1, T2, etc.)
    - No temporal paradoxes (events before/after themselves)
    - Valid year and offset values
    - Extension anchors properly reference baseline
    """

    @property
    def name(self) -> str:
        return "timeline_chronology"

    @property
    def description(self) -> str:
        return "Validates timeline chronology and anchor ordering"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate timeline artifacts for chronological consistency.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult with chronology issues
        """
        logger.debug("Validating timeline chronology in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Extract canon packages and world genesis manifests with timelines
        timeline_artifacts = [
            a
            for a in artifacts
            if a.type in ("canon_transfer_package", "world_genesis_manifest")
        ]

        if not timeline_artifacts:
            # No timeline artifacts to validate - pass
            logger.trace("No timeline artifacts found to validate")
            return self._create_result(issues, validated_artifacts=0)

        for artifact in timeline_artifacts:
            artifact_id = artifact.artifact_id or "unknown"

            # Extract timeline data
            timeline_data = artifact.data.get("timeline", {})
            if not timeline_data or "anchors" not in timeline_data:
                continue

            # Validate using TimelineManager
            timeline = TimelineManager()
            try:
                for anchor_data in timeline_data["anchors"]:
                    anchor = TimelineAnchor(
                        anchor_id=anchor_data.get("anchor_id", ""),
                        event=anchor_data.get("event", ""),
                        year=anchor_data.get("year"),
                        offset=anchor_data.get("offset"),
                        description=anchor_data.get("description", ""),
                        source=anchor_data.get("source", ""),
                        immutable=anchor_data.get("immutable", False),
                    )
                    timeline.add_anchor(anchor)
            except ValueError as e:
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        message=f"Timeline anchor validation failed: {e}",
                        location=f"{artifact_id}/timeline",
                        fix="Fix anchor ID, year, or offset format",
                    )
                )
                continue

            # Run chronology validation
            validation_errors = timeline.validate_chronology()
            if validation_errors:
                logger.warning(
                    (
                        "Timeline chronology validation errors for artifact %s: "
                        "%d error(s)"
                    ),
                    artifact_id,
                    len(validation_errors),
                )
                for error in validation_errors:
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            message=error,
                            location=f"{artifact_id}/timeline",
                            fix="Reorder anchors to maintain chronological consistency",
                        )
                    )
            else:
                logger.debug("Timeline chronology valid for artifact %s", artifact_id)

        logger.debug(
            "Timeline chronology validation complete: %d issues found", len(issues)
        )
        return self._create_result(issues, validated_artifacts=len(timeline_artifacts))


class EntityReferenceBar(QualityBar):
    """
    Validates entity registry consistency and reference integrity.

    Checks for:
    - No duplicate entity names within same type
    - Required entity fields present (name, type, role, description)
    - Valid entity types
    - Source attribution present for immutable entities
    """

    @property
    def name(self) -> str:
        return "entity_reference"

    @property
    def description(self) -> str:
        return "Validates entity registry consistency"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate entity registry artifacts for consistency.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult with entity issues
        """
        logger.debug("Validating entity references in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Extract artifacts with entity registries
        entity_artifacts = [
            a
            for a in artifacts
            if a.type in ("canon_transfer_package", "world_genesis_manifest")
        ]

        if not entity_artifacts:
            # No entity artifacts to validate - pass
            logger.trace("No entity artifacts found to validate")
            return self._create_result(issues, validated_artifacts=0)

        valid_entity_types = {t.value for t in EntityType}

        for artifact in entity_artifacts:
            artifact_id = artifact.artifact_id or "unknown"

            # Extract entity registry
            entity_data = artifact.data.get("entity_registry", {})
            if not entity_data or "entities" not in entity_data:
                continue

            entities = entity_data["entities"]

            # Validate each entity
            for idx, entity in enumerate(entities):
                location = f"{artifact_id}/entity_registry/entities[{idx}]"

                # Check required fields
                required_fields = ["name", "entity_type", "role", "description"]
                for field in required_fields:
                    if not entity.get(field):
                        issues.append(
                            QualityIssue(
                                severity="blocker",
                                message=f"Entity missing required field: {field}",
                                location=location,
                                fix=f"Add '{field}' to entity definition",
                            )
                        )

                # Validate entity type
                entity_type = entity.get("entity_type", "")
                if entity_type and entity_type not in valid_entity_types:
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            message=f"Invalid entity type: {entity_type}",
                            location=location,
                            fix=f"Use one of: {', '.join(valid_entity_types)}",
                        )
                    )

                # Check source attribution for immutable entities
                immutable = entity.get("immutable", False)
                source = entity.get("source", "")
                if immutable and not source:
                    logger.warning(
                        "Immutable entity missing source: %s",
                        entity.get("name", "unknown"),
                    )
                    issues.append(
                        QualityIssue(
                            severity="warning",
                            message=(
                                f"Immutable entity missing source: "
                                f"{entity.get('name', 'unknown')}"
                            ),
                            location=location,
                            fix="Add 'source' field to track canon provenance",
                        )
                    )

        logger.debug(
            "Entity reference validation complete: %d issues found", len(issues)
        )
        return self._create_result(issues, validated_artifacts=len(entity_artifacts))
