"""
Nonlinearity Quality Bar.

Validates that:
- Hubs, loops, and gateways exist where intended
- Multiple meaningful paths present
- Not railroading (choices matter)
- First-choice integrity maintained
"""

import logging
from collections import defaultdict

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class NonlinearityBar(QualityBar):
    """
    Nonlinearity Bar: Hubs/loops/gateways deliberate and meaningful.

    Checks:
    - Planned hubs/loops present in topology
    - Choices lead to different outcomes
    - First-choice integrity (convergence reflects choice)
    - Not railroading (multiple viable paths)
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "nonlinearity"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "Hubs/loops/gateways are deliberate and meaningful"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for nonlinearity.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating nonlinearity in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        sections = [a for a in artifacts if a.type == "manuscript_section"]

        if not sections:
            logger.trace("No manuscript sections found to validate")
            return self._create_result([], sections_checked=0)

        logger.trace(
            "Found %d manuscript sections for nonlinearity validation", len(sections)
        )

        # Build graph for topology analysis
        graph: dict[str, list[str]] = defaultdict(list)
        reverse_graph: dict[str, list[str]] = defaultdict(list)

        for section in sections:
            section_id = section.data.get("id")
            if not section_id:
                continue

            choices = section.data.get("choices", [])
            for choice in choices:
                if isinstance(choice, dict):
                    target = choice.get("target")
                    if target:
                        graph[section_id].append(target)
                        reverse_graph[target].append(section_id)

        # Check for hubs (sections with multiple incoming paths)
        hubs = {
            sid: sources for sid, sources in reverse_graph.items() if len(sources) > 2
        }

        # Check for loops (sections that can reach themselves)
        loops = self._find_loops(graph, sections)

        logger.debug(
            "Topology analysis: %d hubs, %d loops found", len(hubs), len(loops)
        )

        # Check for meaningful choices
        issues.extend(self._check_meaningful_choices(sections, graph))

        # Check first-choice integrity at convergence points
        issues.extend(self._check_convergence_integrity(sections, hubs))

        logger.debug("Nonlinearity validation complete: %d issues found", len(issues))
        return self._create_result(
            issues,
            sections_checked=len(sections),
            hubs_found=len(hubs),
            loops_found=len(loops),
        )

    def _find_loops(
        self, graph: dict[str, list[str]], sections: list[Artifact]
    ) -> list[str]:
        """Find sections that are part of loops."""
        loops: list[str] = []

        for section in sections:
            section_id = section.data.get("id")
            if not section_id:
                continue

            # Simple loop detection: can this section reach itself?
            if self._can_reach(graph, section_id, section_id, set()):
                loops.append(section_id)

        return loops

    def _can_reach(
        self,
        graph: dict[str, list[str]],
        start: str,
        target: str,
        visited: set[str],
    ) -> bool:
        """Check if start can reach target."""
        if start in visited:
            return False

        visited.add(start)

        for neighbor in graph.get(start, []):
            if neighbor == target:
                return True
            if self._can_reach(graph, neighbor, target, visited):
                return True

        # Backtrack: remove from visited to allow other paths to explore
        visited.remove(start)
        return False

    def _check_meaningful_choices(
        self, sections: list[Artifact], graph: dict[str, list[str]]
    ) -> list[QualityIssue]:
        """Check that choices lead to different outcomes."""
        issues: list[QualityIssue] = []

        for section in sections:
            section_id = section.data.get("id", "unknown")
            choices = section.data.get("choices", [])

            # Check for functionally identical choices (same target)
            if len(choices) > 1:
                targets = []
                for choice in choices:
                    if isinstance(choice, dict):
                        targets.append(choice.get("target"))

                # Count duplicate targets
                unique_targets = set(t for t in targets if t)
                if len(unique_targets) < len(choices):
                    issues.append(
                        QualityIssue(
                            severity="warning",
                            message="Multiple choices lead to same immediate target",
                            location=f"section:{section_id}",
                            fix=(
                                "Ensure choices have different immediate "
                                "outcomes or effects"
                            ),
                        )
                    )

        return issues

    def _check_convergence_integrity(
        self, sections: list[Artifact], hubs: dict[str, list[str]]
    ) -> list[QualityIssue]:
        """Check first-choice integrity at convergence points."""
        issues: list[QualityIssue] = []

        # For each hub (convergence point), check if first paragraph reflects entry
        for hub_id, sources in hubs.items():
            # Find the hub section
            hub_section = next(
                (s for s in sections if s.data.get("id") == hub_id), None
            )

            if not hub_section:
                continue

            # Check if text has any variation or conditional content
            text = hub_section.data.get("text", hub_section.data.get("body", ""))

            # Simple heuristic: look for conditional markers
            has_conditional = any(
                marker in text for marker in ["[if ", "[unless ", "{if ", "{{#if"]
            )

            if not has_conditional and len(sources) > 1:
                issues.append(
                    QualityIssue(
                        severity="info",
                        message=(
                            f"Convergence point may lack first-choice integrity "
                            f"(merges {len(sources)} paths)"
                        ),
                        location=f"section:{hub_id}",
                        fix=(
                            "Consider adding variation in first paragraph "
                            "to reflect entering path"
                        ),
                    )
                )

        return issues
