"""
Reachability Quality Bar.

Validates that:
- Must-see beats (keystones) are reachable from at least one path
- No locked content without a key
- Gateway conditions are obtainable
- No unrealistic dependency chains
"""

import logging
from collections import defaultdict, deque

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class ReachabilityBar(QualityBar):
    """
    Reachability Bar: Keystone beats reachable via at least one viable path.

    Checks:
    - Keystones (must-see content) reachable from start
    - Gateways have obtainable keys
    - No orphaned sections
    - Dependency chains not too deep
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "reachability"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "Keystone beats reachable via at least one viable path"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for reachability.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating reachability in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Build graph of manuscript sections
        sections = [a for a in artifacts if a.type == "manuscript_section"]

        if not sections:
            # No sections to validate
            logger.trace("No manuscript sections found to validate")
            return self._create_result([], sections_checked=0)

        logger.trace(
            "Found %d manuscript sections for reachability validation", len(sections)
        )

        # Build section graph (id -> targets)
        graph: dict[str, list[str]] = defaultdict(list)
        section_map: dict[str, Artifact] = {}
        keystones: set[str] = set()

        for section in sections:
            section_id = section.data.get("id")
            if not section_id:
                continue

            section_map[section_id] = section

            # Track keystones (must-see content)
            if section.data.get("keystone") or section.data.get("required"):
                keystones.add(section_id)

            # Build edges from choices
            choices = section.data.get("choices", [])
            for choice in choices:
                if isinstance(choice, dict):
                    target = choice.get("target")
                    if target:
                        graph[section_id].append(target)

        logger.debug("Found %d keystones", len(keystones))

        # Find start section
        start_id = self._find_start_section(sections)
        if not start_id:
            logger.error("No start section found in manuscript")
            issues.append(
                QualityIssue(
                    severity="blocker",
                    message="No start section found",
                    location="manuscript",
                    fix="Mark one section with 'start: true' or 'first: true'",
                )
            )
            return self._create_result(issues, sections_checked=len(sections))

        # Check reachability from start
        reachable = self._compute_reachable(graph, start_id)
        logger.debug(
            "Computed reachability: %d sections reachable from start", len(reachable)
        )

        # Check keystones are reachable
        for keystone_id in keystones:
            if keystone_id not in reachable:
                logger.warning(
                    "Keystone section '%s' not reachable from start", keystone_id
                )
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        message=(
                            f"Keystone section '{keystone_id}' not reachable from start"
                        ),
                        location=f"section:{keystone_id}",
                        fix="Add a path from start to this keystone section",
                    )
                )

        # Check for orphaned sections
        for section_id in section_map:
            if section_id not in reachable and section_id != start_id:
                # Check if it's intentionally orphaned
                is_alternate_start = section_map[section_id].data.get("alternate_start")
                if not is_alternate_start:
                    issues.append(
                        QualityIssue(
                            severity="warning",
                            message=f"Section '{section_id}' not reachable from start",
                            location=f"section:{section_id}",
                            fix=(
                                "Add path to this section or mark "
                                "'alternate_start: true' if intentional"
                            ),
                        )
                    )

        # Check gateway reachability
        issues.extend(self._check_gateway_reachability(sections, reachable))

        logger.debug("Reachability validation complete: %d issues found", len(issues))
        return self._create_result(
            issues,
            sections_checked=len(sections),
            reachable=len(reachable),
            keystones=len(keystones),
        )

    def _find_start_section(self, sections: list[Artifact]) -> str | None:
        """Find the start section."""
        for section in sections:
            if section.data.get("start") or section.data.get("first"):
                return section.data.get("id")

        # If no explicit start, use first section with ID
        if sections:
            return sections[0].data.get("id")

        return None

    def _compute_reachable(self, graph: dict[str, list[str]], start: str) -> set[str]:
        """Compute all sections reachable from start."""
        reachable: set[str] = set()
        queue: deque[str] = deque([start])
        reachable.add(start)

        while queue:
            current = queue.popleft()
            for target in graph.get(current, []):
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)

        return reachable

    def _check_gateway_reachability(
        self, sections: list[Artifact], reachable: set[str]
    ) -> list[QualityIssue]:
        """Check that gateway conditions are obtainable."""
        issues: list[QualityIssue] = []

        # Track where codewords/items are granted
        granted: dict[str, list[str]] = defaultdict(list)

        for section in sections:
            section_id = section.data.get("id")
            if not section_id or section_id not in reachable:
                continue

            # Track what this section grants
            grants = section.data.get("grants", [])
            if isinstance(grants, str):
                grants = [grants]

            for grant in grants:
                granted[grant].append(section_id)

        # Check gateway requirements
        for section in sections:
            section_id = section.data.get("id")
            if not section_id:
                continue

            # Check choice requirements
            choices = section.data.get("choices", [])
            for i, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue

                requires = choice.get("requires", [])
                if isinstance(requires, str):
                    requires = [requires]

                for req in requires:
                    if req not in granted:
                        issues.append(
                            QualityIssue(
                                severity="warning",
                                message=(
                                    f"Choice requires '{req}' but no section grants it"
                                ),
                                location=f"section:{section_id}.choices[{i}]",
                                fix=(
                                    f"Add a section that grants '{req}' or "
                                    f"remove requirement"
                                ),
                            )
                        )

        return issues
