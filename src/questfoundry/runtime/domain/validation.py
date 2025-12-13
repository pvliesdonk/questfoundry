"""Validation for domain cross-references.

This module validates that all references in a loaded Studio are consistent:
- Agent tool_refs point to existing tools
- Agent store access points to existing stores
- Playbook steps reference valid agents
- Knowledge requirements reference existing entries
- etc.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found in the domain."""

    severity: str  # "error" or "warning"
    component_type: str
    component_id: str
    message: str


@dataclass
class ValidationResult:
    """Result of domain validation."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings are ok)."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Just the errors."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Just the warnings."""
        return [i for i in self.issues if i.severity == "warning"]

    def add_error(self, component_type: str, component_id: str, message: str) -> None:
        """Add an error."""
        self.issues.append(
            ValidationIssue("error", component_type, component_id, message)
        )

    def add_warning(self, component_type: str, component_id: str, message: str) -> None:
        """Add a warning."""
        self.issues.append(
            ValidationIssue("warning", component_type, component_id, message)
        )


class DomainValidationError(Exception):
    """Raised when domain validation fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        errors = "\n".join(
            f"  - [{e.component_type}/{e.component_id}] {e.message}"
            for e in result.errors
        )
        super().__init__(f"Domain validation failed:\n{errors}")


def validate_studio(studio: "Studio") -> ValidationResult:  # noqa: F821
    """Validate all cross-references in a loaded studio.

    Args:
        studio: The loaded Studio object

    Returns:
        ValidationResult with any issues found

    Raises:
        DomainValidationError: If validation fails with errors
    """
    from questfoundry.runtime.domain.models import Studio

    assert isinstance(studio, Studio)

    result = ValidationResult()

    # Validate entry agents
    _validate_entry_agents(studio, result)

    # Validate agent references
    _validate_agents(studio, result)

    # Validate playbook references
    _validate_playbooks(studio, result)

    # Validate store references
    _validate_stores(studio, result)

    # Log results
    if result.warnings:
        for w in result.warnings:
            logger.warning(f"[{w.component_type}/{w.component_id}] {w.message}")

    if result.errors:
        for e in result.errors:
            logger.error(f"[{e.component_type}/{e.component_id}] {e.message}")
        raise DomainValidationError(result)

    logger.info("Domain validation passed")
    return result


def _validate_entry_agents(studio: "Studio", result: ValidationResult) -> None:  # noqa: F821
    """Validate entry agent references."""
    for mode, agent_id in studio.entry_agents.items():
        if agent_id not in studio.agents:
            result.add_error(
                "studio",
                studio.id,
                f"Entry agent for '{mode}' references unknown agent: {agent_id}",
            )
        else:
            # Verify the agent is marked as an entry agent
            agent = studio.agents[agent_id]
            if not agent.is_entry_agent:
                result.add_warning(
                    "agent",
                    agent_id,
                    f"Agent is entry_agent for '{mode}' but is_entry_agent=False",
                )


def _validate_agents(studio: "Studio", result: ValidationResult) -> None:  # noqa: F821
    """Validate agent references."""
    for agent_id, agent in studio.agents.items():
        # Validate capability references
        for cap in agent.capabilities:
            if cap.category == "tool" and cap.tool_ref:
                if cap.tool_ref not in studio.tools:
                    result.add_warning(
                        "agent",
                        agent_id,
                        f"Capability '{cap.id}' references unknown tool: {cap.tool_ref}",
                    )

            if cap.category == "store_access" and cap.stores:
                for store_id in cap.stores:
                    if store_id not in studio.stores:
                        result.add_error(
                            "agent",
                            agent_id,
                            f"Capability '{cap.id}' references unknown store: {store_id}",
                        )

            if cap.category == "artifact_action" and cap.artifact_types:
                for art_type in cap.artifact_types:
                    if (
                        art_type not in studio.artifact_types
                        and art_type not in studio.asset_types
                    ):
                        result.add_warning(
                            "agent",
                            agent_id,
                            f"Capability '{cap.id}' references unknown artifact type: {art_type}",
                        )

        # Validate knowledge requirements
        kr = agent.knowledge_requirements
        all_knowledge_refs = kr.must_know + kr.role_specific + kr.can_lookup
        for entry_id in all_knowledge_refs:
            if entry_id not in studio.knowledge_entries:
                result.add_warning(
                    "agent",
                    agent_id,
                    f"Knowledge requirement references unknown entry: {entry_id}",
                )


def _validate_playbooks(studio: "Studio", result: ValidationResult) -> None:  # noqa: F821
    """Validate playbook references."""
    for playbook_id, playbook in studio.playbooks.items():
        # Validate entry phase exists
        if playbook.entry_phase not in playbook.phases:
            result.add_error(
                "playbook",
                playbook_id,
                f"Entry phase '{playbook.entry_phase}' not found in phases",
            )

        # Validate phase references
        for phase_id, phase in playbook.phases.items():
            # Validate depends_on
            for dep in phase.depends_on:
                if dep not in playbook.phases:
                    result.add_error(
                        "playbook",
                        playbook_id,
                        f"Phase '{phase_id}' depends on unknown phase: {dep}",
                    )

            # Validate on_success/on_failure transitions
            for transition, name in [
                (phase.on_success, "on_success"),
                (phase.on_failure, "on_failure"),
            ]:
                if transition and transition.next_phases:
                    for next_phase in transition.next_phases:
                        if next_phase not in playbook.phases:
                            result.add_error(
                                "playbook",
                                playbook_id,
                                f"Phase '{phase_id}' {name} references unknown phase: {next_phase}",
                            )

            # Validate step references
            for step_id, step in phase.steps.items():
                # Validate depends_on within steps
                for dep in step.depends_on:
                    if dep not in phase.steps:
                        result.add_error(
                            "playbook",
                            playbook_id,
                            f"Step '{step_id}' in phase '{phase_id}' depends on unknown step: {dep}",
                        )

                # Validate agent references
                if step.specific_agent and step.specific_agent not in studio.agents:
                    result.add_warning(
                        "playbook",
                        playbook_id,
                        f"Step '{step_id}' references unknown agent: {step.specific_agent}",
                    )

        # Validate quality criteria
        for qc_id in playbook.quality_criteria:
            if qc_id not in studio.quality_criteria:
                result.add_warning(
                    "playbook",
                    playbook_id,
                    f"References unknown quality criteria: {qc_id}",
                )


def _validate_stores(studio: "Studio", result: ValidationResult) -> None:  # noqa: F821
    """Validate store references."""
    for store_id, store in studio.stores.items():
        for art_type in store.artifact_types:
            if (
                art_type not in studio.artifact_types
                and art_type not in studio.asset_types
            ):
                result.add_warning(
                    "store",
                    store_id,
                    f"References unknown artifact type: {art_type}",
                )
